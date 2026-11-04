"""Run a command with a restricted view of the filesystem (standalone, standard library only).

    python3 sandbox_exec.py [--hide ROOT]... [--ro PATH]... [--rw PATH]... [--mask PATH]...
                            [--ro-file SRC DST]... [--tmp DIR] [--home DIR] [--cwd DIR] [--newpid]
                            -- CMD [ARG]...

The command runs in new user and mount namespaces (no privileges needed; the kernel must allow
unprivileged user namespaces). Every `--hide` root (default /lustre and /home) is covered by an
empty tmpfs, then each `--ro` / `--rw` path is mounted back at its own location (and at its
resolved location, if it is reached through symlinks). `--mask` covers a path inside an allowed
tree. `--ro-file` shows the regular file SRC, read-only, in place of the existing regular file DST
inside a read-write tree; both are given as normalized absolute paths without a symlink anywhere
along them, and are held open while they are checked and mounted. `--tmp` replaces /tmp,
`--home` sets HOME. With `--newpid` the command also gets its own PID namespace and /proc, so it
cannot see other processes, and every process it leaves behind is stopped when it exits. The uid
and gid are unchanged and no capability survives the exec.

The file is copied into job scripts and containers by path, so it must not import rsibench.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import resource
import signal
import stat
import sys
import tempfile

CLONE_NEWNS, CLONE_NEWUSER, CLONE_NEWPID = 0x00020000, 0x10000000, 0x20000000
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC = 1, 2, 4, 8
MS_REMOUNT, MS_BIND, MS_REC, MS_PRIVATE = 32, 4096, 16384, 1 << 18
MS_NOATIME, MS_NODIRATIME, MS_RELATIME = 1024, 2048, 1 << 21
MNT_DETACH = 2
# statvfs f_flag bits -> mount flags that are locked on a bind mount and must be kept on remount
_LOCKED = {2: MS_NOSUID, 4: MS_NODEV, 8: MS_NOEXEC, 1024: MS_NOATIME, 2048: MS_NODIRATIME,
           4096: MS_RELATIME}

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong,
                        ctypes.c_void_p]
_libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
_libc.unshare.argtypes = [ctypes.c_int]
_libc.syscall.restype = ctypes.c_long
SYS_OPENAT2, AT_FDCWD, RESOLVE_NO_SYMLINKS = 437, -100, 0x04
O_PATH = getattr(os, "O_PATH", 0o10000000)
_use_openat2 = True  # tests turn it off to exercise the component-by-component fallback


class _OpenHow(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("mode", ctypes.c_uint64),
                ("resolve", ctypes.c_uint64)]


class SandboxError(RuntimeError):
    pass


def _check(ret: int, what: str) -> None:
    if ret != 0:
        err = ctypes.get_errno()
        raise SandboxError(f"{what}: {os.strerror(err)}")


def _mount(src, dst, fstype, flags, data=None) -> None:
    enc = lambda s: s.encode() if isinstance(s, str) else s  # noqa: E731
    _check(_libc.mount(enc(src), enc(dst), enc(fstype), flags, enc(data)),
           f"mount {src} -> {dst} ({fstype}, {flags:#x})")


def _write(path: str, text: str) -> None:
    with open(path, "w") as fh:
        fh.write(text)


def _ensure(target: str, is_dir: bool) -> None:
    if is_dir:
        os.makedirs(target, exist_ok=True)
    elif not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "a").close()


def _bind(src: str, dst: str, readonly: bool) -> None:
    _mount(src, dst, None, MS_BIND | MS_REC)
    if readonly:
        locked = 0
        flags = os.statvfs(dst).f_flag
        for bit, ms in _LOCKED.items():
            if flags & bit:
                locked |= ms
        _mount(None, dst, None, MS_REMOUNT | MS_BIND | MS_RDONLY | locked)


def _under(path: str, roots: list[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)


def open_no_symlinks(path: str) -> int:
    """An O_PATH descriptor of the regular file `path`, refusing a symlink in any component (so
    a concurrent swap cannot redirect it): openat2 with RESOLVE_NO_SYMLINKS, else a walk that
    opens one component at a time with O_NOFOLLOW."""
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise SandboxError(f"not a normalized absolute path: {path}")
    flags = O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = -1
    if _use_openat2:
        how = _OpenHow(flags, 0, RESOLVE_NO_SYMLINKS)
        fd = _libc.syscall(ctypes.c_long(SYS_OPENAT2), ctypes.c_int(AT_FDCWD), path.encode(),
                           ctypes.byref(how), ctypes.c_size_t(ctypes.sizeof(how)))
        err = ctypes.get_errno() if fd < 0 else 0
        if err == errno.ELOOP:
            raise SandboxError(f"{path} goes through a symlink")
        if fd < 0 and err not in (errno.ENOSYS, errno.EPERM):  # EPERM: a seccomp filter
            raise SandboxError(f"{path}: {os.strerror(err)}")
    if fd < 0:
        fd = os.open("/", O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
        parts = path.strip("/").split("/")
        try:
            for i, part in enumerate(parts):
                # on a symlink, O_PATH|O_NOFOLLOW opens the link itself: O_DIRECTORY then fails
                directory = os.O_DIRECTORY if i < len(parts) - 1 else 0
                try:
                    child = os.open(part, flags | directory, dir_fd=fd)
                except OSError as exc:
                    raise SandboxError(f"{path} goes through a symlink or a missing "
                                       f"directory: {exc.strerror}") from exc
                os.close(fd)
                fd = child
        except BaseException:
            os.close(fd)
            raise
    mode = os.fstat(fd).st_mode
    if not stat.S_ISREG(mode):
        os.close(fd)
        kind = "a symlink" if stat.S_ISLNK(mode) else "not a regular file"
        raise SandboxError(f"{path} is {kind}")
    return fd


def _bind_file(src_fd: int, dst: str) -> None:
    """Mount the pinned file src_fd read-only over the regular file dst (already checked to be
    inside a read-write tree), through descriptors so that no path is resolved twice."""
    dst_fd = open_no_symlinks(dst)
    try:
        _mount(f"/proc/self/fd/{src_fd}", f"/proc/self/fd/{dst_fd}", None, MS_BIND)
    finally:
        os.close(dst_fd)
    top = open_no_symlinks(dst)  # the new mount, if nothing moved: it shows the source file
    try:
        shown, source = os.fstat(top), os.fstat(src_fd)
        if (shown.st_dev, shown.st_ino) != (source.st_dev, source.st_ino):
            raise SandboxError(f"{dst} changed while it was mounted")
        target = f"/proc/self/fd/{top}"
        flags = os.statvfs(target).f_flag
        locked = 0
        for bit, ms in _LOCKED.items():
            if flags & bit:
                locked |= ms
        _mount(None, target, None, MS_REMOUNT | MS_BIND | MS_RDONLY | locked)
    finally:
        os.close(top)


def setup(hide: list[str], ro: list[str], rw: list[str], mask: list[str], tmp: str | None,
          newpid: bool, ro_files: list | None = None) -> None:
    """Enter the namespaces and arrange the mounts (call in the process that will exec)."""
    uid, gid = os.getuid(), os.getgid()
    flags = CLONE_NEWUSER | CLONE_NEWNS | (CLONE_NEWPID if newpid else 0)
    _check(_libc.unshare(flags), "unshare (are unprivileged user namespaces allowed?)")
    _write("/proc/self/setgroups", "deny")
    _write("/proc/self/uid_map", f"{uid} {uid} 1\n")
    _write("/proc/self/gid_map", f"{gid} {gid} 1\n")
    _mount(None, "/", None, MS_REC | MS_PRIVATE)

    hide = [os.path.abspath(h) for h in hide if os.path.isdir(h)]
    allowed = []  # (source, [targets], readonly, is_dir)
    for paths, readonly in ((ro, True), (rw, False)):
        for p in paths:
            if not os.path.exists(p):
                raise SandboxError(f"allowed path does not exist: {p}")
            src = os.path.realpath(p)
            targets = sorted({os.path.abspath(p), src})
            allowed.append((src, targets, readonly, os.path.isdir(src)))
    if tmp and "/tmp" not in hide:
        # (when /tmp is itself a hidden root it is already a private tmpfs; binding a directory
        # that lives in an allowed tree over it would nest that tree inside itself)
        allowed.append((os.path.realpath(tmp), ["/tmp"], False, True))

    # Stage the allowed trees outside the hidden roots before covering them.
    stage = tempfile.mkdtemp(prefix="rsib-sb-", dir="/dev/shm")
    _mount("tmpfs", stage, "tmpfs", MS_NOSUID | MS_NODEV, "mode=700")
    staged = []
    for i, (src, targets, readonly, is_dir) in enumerate(allowed):
        s = os.path.join(stage, str(i))
        _ensure(s, is_dir)
        _mount(src, s, None, MS_BIND | MS_REC)
        staged.append(s)
    # Pinned before the hidden roots are covered; mounted from the descriptor afterwards.
    writable = [t for _, targets, readonly, _ in allowed if not readonly for t in targets]
    files = []  # (source descriptor, destination)
    for src, dst in ro_files or []:
        if not _under(os.path.normpath(dst), writable):
            raise SandboxError(f"--ro-file destination is outside the read-write trees: {dst}")
        files.append((open_no_symlinks(src), dst))

    for h in hide:
        _mount("tmpfs", h, "tmpfs", MS_NOSUID | MS_NODEV, "mode=755")

    # Real locations first, parents before children, so a read-write tree can sit inside a
    # read-only one. Then the paths that reach them through symlinks: mounting on a symlink
    # follows it, so an alias that already leads to its source is left as it is, and one whose
    # link leads through another alias waits until that alias is mounted.
    def binds(aliases: bool) -> list[tuple[str, int]]:
        return sorted(((t, i) for i, (src, targets, _, _) in enumerate(allowed)
                       for t in targets if (t != src) == aliases), key=lambda b: len(b[0]))

    for t, i in binds(False):
        _, _, readonly, is_dir = allowed[i]
        _ensure(t, is_dir)
        _bind(staged[i], t, readonly)
    pending = binds(True)
    while pending:
        waiting = []
        for t, i in pending:
            _, _, readonly, is_dir = allowed[i]
            if not _under(t, hide) and t != "/tmp":
                continue  # an alias outside the hidden roots resolves by itself
            if os.path.islink(t) and not os.path.exists(t):
                waiting.append((t, i))
                continue
            if os.path.exists(t) and os.path.samefile(t, staged[i]):
                continue
            _ensure(t, is_dir)
            _bind(staged[i], t, readonly)
        if len(waiting) == len(pending):
            break  # links into nothing that is allowed: they stay dangling
        pending = waiting
    for src_fd, dst in files:
        try:
            _bind_file(src_fd, dst)
        finally:
            os.close(src_fd)
    for m in mask:
        if os.path.isdir(m):
            _mount("tmpfs", m, "tmpfs", MS_NOSUID | MS_NODEV, "mode=755")
        elif os.path.exists(m):
            _bind("/dev/null", m, True)
    _check(_libc.umount2(stage.encode(), MNT_DETACH), "umount stage")
    try:
        os.rmdir(stage)
    except OSError:
        pass


def _run_as_init(argv: list[str], env: dict, cwd: str | None) -> int:
    """After unshare(CLONE_NEWPID) the next child is PID 1 of the new namespace."""
    pid = os.fork()
    if pid == 0:
        try:
            _mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
            if cwd:
                os.chdir(cwd)
            os.execvpe(argv[0], argv, env)
        except Exception as exc:  # noqa: BLE001
            print(f"sandbox: {exc}", file=sys.stderr)
            os._exit(127)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
        signal.signal(sig, lambda s, _f: os.kill(pid, s))
    while True:
        try:
            _, status = os.waitpid(pid, 0)
            break
        except InterruptedError:
            continue
    return os.waitstatus_to_exitcode(status)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        print("usage: sandbox_exec.py [options] -- CMD [ARG]...", file=sys.stderr)
        return 2
    split = argv.index("--")
    ap = argparse.ArgumentParser(prog="sandbox_exec.py")
    ap.add_argument("--hide", action="append", default=None)
    ap.add_argument("--ro", action="append", default=[])
    ap.add_argument("--rw", action="append", default=[])
    ap.add_argument("--mask", action="append", default=[])
    ap.add_argument("--ro-file", action="append", default=[], nargs=2, metavar=("SRC", "DST"))
    ap.add_argument("--tmp")
    ap.add_argument("--home")
    ap.add_argument("--cwd")
    ap.add_argument("--newpid", action="store_true")
    ap.add_argument("--max-procs", type=int, default=0)
    args = ap.parse_args(argv[:split])
    cmd = argv[split + 1:]
    if not cmd:
        print("sandbox_exec.py: no command", file=sys.stderr)
        return 2
    hide = args.hide if args.hide is not None else ["/lustre", "/home"]
    env = dict(os.environ)
    if args.home:
        env["HOME"] = args.home
        if args.home not in args.rw:
            args.rw.append(args.home)
    if args.tmp:
        env["TMPDIR"] = "/tmp"
    start_dir = args.cwd or os.getcwd()
    try:
        setup(hide, args.ro, args.rw, args.mask, args.tmp, args.newpid, args.ro_file)
    except (SandboxError, OSError) as exc:
        print(f"sandbox: {exc}", file=sys.stderr)
        return 126
    # The inherited working directory still points into the unrestricted tree (`cd ..` would
    # walk out of the sandbox); enter it again by path, through the new mounts.
    try:
        os.chdir(start_dir)
    except OSError:
        os.chdir("/")
    if args.max_procs > 0:
        # Counted in this sandbox's user namespace and inherited by everything it starts, also
        # by processes that leave the process group: forking past it fails with EAGAIN.
        resource.setrlimit(resource.RLIMIT_NPROC, (args.max_procs, args.max_procs))
    if args.newpid:
        return _run_as_init(cmd, env, None)
    os.execvpe(cmd[0], cmd, env)
    return 127  # not reached


if __name__ == "__main__":
    sys.exit(main())
