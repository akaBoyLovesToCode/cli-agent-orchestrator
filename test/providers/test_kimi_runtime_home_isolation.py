"""Security isolation tests for the per-worker Kimi runtime home.

**Finding 1 (P2).** ``_copy_tree`` passes ``symlinks=True`` so an *internal*
symlink under a secret directory (see :data:`SECRET_DIR_NAMES`) was reproduced
verbatim into the runtime home. Writing through the runtime copy then wrote
through the link back into shared/source state. A copied secret tree must never
contain a writable path back out of the runtime home, so the secret copy policy
is deliberately not the ordinary ``skills``/``plugins`` policy. The policy is
now covered by exercising :meth:`KimiCodeRuntimeHomeBuilder._copy_secret_tree`
directly: as of Step 4F no preserved directory is secret any more, because
``credentials/`` is *shared* rather than copied (see :data:`SHARED_AUTH_DIRS`),
and the guard is what keeps a future secret entry in :data:`PRESERVE_DIRS`
safe.

**Finding 2 (P3).** ``_copy_trust_tree`` did ``entries = sorted(scan, ...)``,
which materialised the *entire* source directory before :data:`MAX_TRUST_ENTRIES`
was applied. The record count was bounded but the enumeration, allocation and
scandir consumption were not.

**Step 4F — shared OAuth state.** Kimi Code rotates OAuth refresh tokens on use
and serialises cross-process refreshes through a ``proper-lockfile`` lock under
``oauth/``; both the credential store (``credentials/<name>.json``) and the lock
domain (``oauth/<name>.lock``) derive from ``KIMI_CODE_HOME``. Snapshot-copying
``credentials/`` into a disposable runtime home therefore stranded the rotated
token in the worker and left the operator's store holding a stale,
server-invalidated refresh token — which the next standalone refresh turned
into a revoked tombstone ("Stored token ... was rejected; re-login required").
``credentials/`` and ``oauth/`` are now directory *symlinks* to the source home:
the one deliberate exception to the no-escape invariant, scoped to exactly the
two auth paths that must share one authoritative lineage. Everything else in
the runtime home still satisfies the no-escape invariant.

All tests use ``tmp_path`` only; the real ``~/.kimi-code`` is never touched.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict

from cli_agent_orchestrator.providers import kimi_runtime_home as mod
from cli_agent_orchestrator.providers.kimi_runtime_home import (
    MAX_TRUST_ENTRIES,
    TRUST_DIR_NAME,
    KimiCodeRuntimeHomeBuilder,
)


def _build(source: Path, temp: Path):
    return KimiCodeRuntimeHomeBuilder(source, temp).build()


def _assert_no_escape(root: Path) -> None:
    """Every entry under ``root`` must be a real entry inside ``root``.

    This is the core credential invariant: no symlink, and nothing whose real
    path resolves outside the runtime home (i.e. no writable path back into the
    source home or any shared target).
    """

    assert not root.is_symlink(), f"{root} is a symlink"
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        assert not path.is_symlink(), f"{path} is a symlink"
        real = Path(os.path.realpath(path))
        assert real.is_relative_to(resolved_root), f"{path} escapes to {real}"


def _make_auth_source(source: Path, token: str = "R1") -> Path:
    """A source home holding a synthetic OAuth credential and lock sentinel."""

    creds = source / "credentials"
    creds.mkdir(parents=True)
    (creds / "kimi-code.json").write_text(
        json.dumps({"accessToken": "A1", "refreshToken": token}), encoding="utf-8"
    )
    oauth = source / "oauth"
    oauth.mkdir()
    (oauth / "kimi-code").write_text("", encoding="utf-8")
    return source


class TestSharedAuthState:
    """Step 4F — ``credentials/`` and ``oauth/`` are shared, never snapshotted.

    Kimi's own multi-process coordination (a ``proper-lockfile`` lock plus a
    re-read-and-adopt after lock acquisition) only works when every process
    that authenticates as the same user resolves the *same* credential file and
    the *same* lock domain. Per-worker copies gave each worker a private lock
    domain and a private credential generation, which is exactly the shape that
    produced the production "re-login required" rejections.
    """

    def test_credentials_and_oauth_are_directory_links_to_the_source(self, tmp_path):
        source = _make_auth_source(tmp_path / "src")

        result = _build(source, tmp_path / "temp")

        creds_link = result.home / "credentials"
        oauth_link = result.home / "oauth"
        assert creds_link.is_symlink()
        assert oauth_link.is_symlink()
        assert Path(os.path.realpath(creds_link)) == Path(os.path.realpath(source / "credentials"))
        assert Path(os.path.realpath(oauth_link)) == Path(os.path.realpath(source / "oauth"))
        assert result.shared_auth_dirs == ["credentials", "oauth"]
        # The worker reads the operator's current credential through the link.
        assert json.loads((creds_link / "kimi-code.json").read_text())["refreshToken"] == "R1"

    def test_atomic_rotation_through_the_worker_is_visible_at_the_source(self, tmp_path):
        """The exact Step 4F rotation: R1 -> R2 must land in the shared store.

        Kimi's ``FileTokenStorage.save`` writes ``<name>.json.tmp.<pid>.<rand>``
        next to the target and atomically renames it over ``<name>.json``.
        Through a *directory* link that dance happens inside the shared store,
        so the link survives and every reader sees the new generation. (A
        file-level link would instead be *replaced* by the rename, silently
        re-splitting the lineage — that is why the link is at directory level.)
        """

        source = _make_auth_source(tmp_path / "src")

        result = _build(source, tmp_path / "temp")
        worker_creds = result.home / "credentials"

        # Simulate Kimi's atomic save through the worker-visible path.
        tmp = worker_creds / "kimi-code.json.tmp.1234.abcd"
        tmp.write_text(json.dumps({"accessToken": "A2", "refreshToken": "R2"}), encoding="utf-8")
        os.rename(tmp, worker_creds / "kimi-code.json")

        assert (
            json.loads((source / "credentials" / "kimi-code.json").read_text())["refreshToken"]
            == "R2"
        )
        assert worker_creds.is_symlink(), "the rename must not replace the directory link"
        # A second worker home observes the same generation.
        other = _build(source, tmp_path / "temp2")
        assert (
            json.loads((other.home / "credentials" / "kimi-code.json").read_text())["refreshToken"]
            == "R2"
        )

    def test_two_workers_share_one_lock_domain(self, tmp_path):
        source = _make_auth_source(tmp_path / "src")

        first = _build(source, tmp_path / "temp1")
        second = _build(source, tmp_path / "temp2")

        sentinel_a = first.home / "oauth" / "kimi-code"
        sentinel_b = second.home / "oauth" / "kimi-code"
        assert os.path.realpath(sentinel_a) == os.path.realpath(sentinel_b)
        # A lock directory taken through one worker's path is visible through the
        # other's: one coordination domain, not two.
        lock_a = first.home / "oauth" / "kimi-code.lock"
        lock_a.mkdir()
        assert (second.home / "oauth" / "kimi-code.lock").is_dir()
        assert (source / "oauth" / "kimi-code.lock").is_dir()

    def test_cleanup_removes_only_the_links_never_the_shared_state(self, tmp_path):
        source = _make_auth_source(tmp_path / "src")
        builder = KimiCodeRuntimeHomeBuilder(source, tmp_path / "temp")
        result = builder.build()
        assert (result.home / "credentials").is_symlink()

        assert builder.cleanup() is True

        assert not result.home.exists()
        assert (
            json.loads((source / "credentials" / "kimi-code.json").read_text())["refreshToken"]
            == "R1"
        )
        assert (source / "oauth" / "kimi-code").is_file()

    def test_rmtree_of_the_terminal_dir_does_not_follow_auth_links(self, tmp_path):
        """The kimi_cli reset path ``shutil.rmtree(terminal_dir)`` must unlink
        the auth links, never delete through them into the operator's store."""

        source = _make_auth_source(tmp_path / "src")
        terminal_dir = tmp_path / "terminal"
        result = _build(source, terminal_dir)

        shutil.rmtree(terminal_dir)

        assert not terminal_dir.exists()
        assert (
            json.loads((source / "credentials" / "kimi-code.json").read_text())["refreshToken"]
            == "R1"
        )
        assert (source / "oauth").is_dir()
        assert result.home  # silence unused-result lint; the build succeeded

    def test_absent_source_auth_state_links_nothing(self, tmp_path):
        """A logged-out source home yields no auth entries and no crash."""

        source = tmp_path / "src"
        source.mkdir()

        result = _build(source, tmp_path / "temp")

        assert result.shared_auth_dirs == []
        assert not (result.home / "credentials").exists()
        assert not (result.home / "oauth").exists()
        # A logged-out home is never mutated: no oauth directory is synthesised.
        assert not (source / "oauth").exists()

    def test_source_oauth_dir_is_created_when_credentials_are_shared(self, tmp_path):
        """Kimi creates ``oauth/`` on first refresh; creating it here keeps the
        worker in the shared lock domain from its very first refresh instead of
        letting it build a private one inside the disposable home."""

        source = tmp_path / "src"
        creds = source / "credentials"
        creds.mkdir(parents=True)
        (creds / "kimi-code.json").write_text("{}", encoding="utf-8")

        result = _build(source, tmp_path / "temp")

        assert (source / "oauth").is_dir()
        assert stat.S_IMODE(os.stat(source / "oauth").st_mode) == 0o700
        link = result.home / "oauth"
        assert link.is_symlink()
        assert Path(os.path.realpath(link)) == Path(os.path.realpath(source / "oauth"))
        assert result.shared_auth_dirs == ["credentials", "oauth"]

    def test_a_regular_file_named_credentials_fails_closed(self, tmp_path, caplog):
        """A non-directory ``credentials`` is pathological: skip it, keep the
        build alive, and let Kimi's own auth error surface in the worker."""

        source = tmp_path / "src"
        source.mkdir()
        (source / "credentials").write_text("not-a-directory")

        result = _build(source, tmp_path / "temp")

        assert result.shared_auth_dirs == []
        assert not (result.home / "credentials").exists()
        assert not (source / "oauth").exists(), "no credentials shared -> no oauth dir created"

    def test_a_dangling_source_credentials_symlink_fails_closed(self, tmp_path, caplog):
        source = tmp_path / "src"
        source.mkdir()
        (source / "credentials").symlink_to(source / "gone", target_is_directory=True)

        result = _build(source, tmp_path / "temp")

        assert result.shared_auth_dirs == []
        assert not os.path.lexists(result.home / "credentials")

    def test_a_symlinked_source_credentials_dir_is_shared_at_its_own_path(self, tmp_path):
        """An operator who links ``credentials`` elsewhere has arranged their own
        authoritative store; the worker links the same path (link-to-link, the
        documented ``bin/`` policy) and resolution lands on the real store."""

        source = tmp_path / "src"
        source.mkdir()
        real = tmp_path / "real-creds"
        real.mkdir()
        (real / "kimi-code.json").write_text('{"refreshToken": "R1"}', encoding="utf-8")
        (source / "credentials").symlink_to(real, target_is_directory=True)

        result = _build(source, tmp_path / "temp")

        link = result.home / "credentials"
        assert link.is_symlink()
        assert json.loads((link / "kimi-code.json").read_text())["refreshToken"] == "R1"
        assert result.shared_auth_dirs == ["credentials", "oauth"]

    def test_the_build_never_mutates_source_permissions_or_content(self, tmp_path):
        source = _make_auth_source(tmp_path / "src")

        def snapshot(root: Path) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for path in sorted(root.rglob("*")):
                rel = str(path.relative_to(root))
                if path.is_symlink():
                    out[rel] = ("link", os.readlink(path))
                elif path.is_dir():
                    out[rel] = ("dir", stat.S_IMODE(os.stat(path).st_mode))
                else:
                    out[rel] = ("file", path.read_bytes())
            return out

        before = snapshot(source)

        _build(source, tmp_path / "temp")

        assert snapshot(source) == before

    def test_mcp_json_and_runtime_state_stay_isolated(self, tmp_path):
        """Sharing auth state must not widen sharing beyond the two auth paths."""

        source = _make_auth_source(tmp_path / "src")
        (source / "mcp.json").write_text('{"mcpServers": {"user": {"command": "x"}}}')
        (source / "sessions").mkdir()
        (source / "sessions" / "s.json").write_text("{}")

        result = _build(source, tmp_path / "temp")

        mcp = result.home / "mcp.json"
        assert mcp.is_file() and not mcp.is_symlink()
        assert not (result.home / "sessions").exists()
        # Writes to the worker's mcp.json cannot reach the source file.
        before = (source / "mcp.json").read_text()
        mcp.write_text('{"mcpServers": {}}')
        assert (source / "mcp.json").read_text() == before


class TestSecretTreeCopyPolicy:
    """Finding 1 (P2) — the secret copy policy keeps no writable path out.

    Since Step 4F no preserved directory is secret, so nothing routes through
    :meth:`KimiCodeRuntimeHomeBuilder._copy_secret_tree` by default. The policy
    remains the guard that makes a *future* secret entry in
    :data:`PRESERVE_DIRS` safe, so it is exercised directly here.
    """

    @staticmethod
    def _copy(src: Path, dst: Path) -> None:
        KimiCodeRuntimeHomeBuilder._copy_secret_tree(src, dst)

    def test_internal_symlinks_are_not_reproduced(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "real.json").write_text("real")
        (src / "alias.json").symlink_to("real.json")
        deep = tmp_path / "external" / "deep"
        deep.mkdir(parents=True)
        (deep / "token.json").write_text("external-token")
        (src / "linked-dir").symlink_to(tmp_path / "external", target_is_directory=True)

        dst = tmp_path / "dst"
        self._copy(src, dst)

        _assert_no_escape(dst)
        assert (dst / "real.json").read_text() == "real"
        assert not (dst / "alias.json").exists()
        assert not (dst / "linked-dir").exists()

    def test_ordinary_files_are_real_private_copies(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        plain = src / "plain.json"
        plain.write_text("plain")
        os.chmod(plain, 0o644)

        dst = tmp_path / "dst"
        self._copy(src, dst)

        copied = dst / "plain.json"
        assert copied.is_file() and not copied.is_symlink()
        assert copied.read_text() == "plain"
        assert stat.S_IMODE(os.stat(copied).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(dst).st_mode) == 0o700

    def test_writing_the_copy_cannot_mutate_the_source(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "plain.json").write_text("plain-original")

        dst = tmp_path / "dst"
        self._copy(src, dst)
        (dst / "plain.json").write_text("tampered")

        assert (src / "plain.json").read_text() == "plain-original"


class _CountingScandir:
    """Wrap a real ``os.scandir`` result and count pulled entries."""

    def __init__(self, inner, counter: Dict[str, int]) -> None:
        self._inner = inner
        self._iter = iter(inner)
        self._counter = counter

    def __iter__(self) -> "_CountingScandir":
        return self

    def __next__(self):
        entry = next(self._iter)
        self._counter["pulled"] += 1
        return entry

    def __enter__(self) -> "_CountingScandir":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self._inner.close()


class TestTrustTraversalBudget:
    """Finding 2 — the budget must bound enumeration, not just processing."""

    def test_scandir_iterator_is_not_drained_past_the_budget(self, tmp_path, monkeypatch):
        """The exact reproduction: 10,000 entries, budget 4,096.

        Pre-fix the record count was bounded but ``sorted(scan)`` drained the
        whole directory, consuming all 10,000 iterator entries. The iterator
        itself must stop at the budget.
        """

        source = tmp_path / "src"
        source.mkdir()
        trust = source / TRUST_DIR_NAME
        trust.mkdir()
        total = 10_000
        for i in range(total):
            (trust / f"wd_{i:05d}").write_text("{}")

        counter = {"pulled": 0}
        real_scandir = os.scandir

        def counting_scandir(path, *args, **kwargs):
            return _CountingScandir(real_scandir(path, *args, **kwargs), counter)

        monkeypatch.setattr(mod.os, "scandir", counting_scandir)

        result = _build(source, tmp_path / "temp")

        assert result.trust_truncated is True
        assert len(result.trust_records) == MAX_TRUST_ENTRIES
        # The bound is the budget plus a single look-ahead entry used only to
        # detect that more work remains; it is emphatically not `total`.
        assert counter["pulled"] <= MAX_TRUST_ENTRIES + 1, counter["pulled"]

    def test_many_entries_are_not_fully_enumerated(self, tmp_path, monkeypatch):
        source = tmp_path / "src"
        source.mkdir()
        trust = source / TRUST_DIR_NAME
        trust.mkdir()
        total = MAX_TRUST_ENTRIES * 3 + 7
        for i in range(total):
            (trust / f"d{i:05d}").mkdir()

        counter: Dict[str, int] = {"pulled": 0}
        real_scandir = os.scandir

        def counting_scandir(path, *args, **kwargs):
            return _CountingScandir(real_scandir(path, *args, **kwargs), counter)

        monkeypatch.setattr(mod.os, "scandir", counting_scandir)

        result = _build(source, tmp_path / "temp")

        assert result.trust_truncated is True
        assert counter["pulled"] <= MAX_TRUST_ENTRIES + 1, counter["pulled"]

    def test_exact_budget_is_not_reported_as_truncated(self, tmp_path, monkeypatch):
        """Boundary preserved: exactly MAX entries is complete, not truncated."""

        source = tmp_path / "src"
        source.mkdir()
        trust = source / TRUST_DIR_NAME
        trust.mkdir()
        for i in range(MAX_TRUST_ENTRIES):
            (trust / f"f{i:05d}").write_text("{}")

        counter: Dict[str, int] = {"pulled": 0}
        real_scandir = os.scandir

        def counting_scandir(path, *args, **kwargs):
            return _CountingScandir(real_scandir(path, *args, **kwargs), counter)

        monkeypatch.setattr(mod.os, "scandir", counting_scandir)

        result = _build(source, tmp_path / "temp")

        assert result.trust_truncated is False
        assert len(result.trust_records) == MAX_TRUST_ENTRIES
        assert counter["pulled"] == MAX_TRUST_ENTRIES


class TestPreservedTreeSymlinkPolicy:
    """Finding 3 (P2) — a preserved tree keeps a user's links *and* their meaning.

    ``skills/`` and ``plugins/`` are ordinary preserved trees, so their symlinks
    are user semantics and are reproduced verbatim. That is correct only while the
    link still names the same target after the tree moves: the runtime home is a
    different directory, so a *relative* link that left the source home left the
    runtime home too and dangled — reproduced, a linked shared skills directory
    existed in the real home and did not exist for the worker. Absolute links are
    stable, and relative links inside the tree keep their text because the
    preserved structure resolves them the same way.
    """

    def test_external_relative_link_keeps_its_target(self, tmp_path):
        """The linked-in skill is readable from the runtime home."""

        source = tmp_path / "src"
        skills = source / "skills"
        skills.mkdir(parents=True)
        shared = tmp_path / "shared-skills"
        shared.mkdir()
        (shared / "SKILL.md").write_text("shared skill")
        # A link that leaves the source home, exactly as an operator would make it.
        (skills / "shared").symlink_to(os.path.relpath(shared, skills))

        result = _build(source, tmp_path / "temp")

        link = result.home / "skills" / "shared"
        assert link.is_symlink()
        assert link.exists(), "the linked skill must still resolve in the runtime home"
        assert (link / "SKILL.md").read_text() == "shared skill"

    def test_internal_relative_link_is_preserved_verbatim(self, tmp_path):
        """A link inside the tree keeps its relative text and still resolves."""

        source = tmp_path / "src"
        skills = source / "skills"
        skills.mkdir(parents=True)
        (skills / "SKILL.md").write_text("internal skill")
        (skills / "alias").symlink_to("SKILL.md")

        result = _build(source, tmp_path / "temp")

        link = result.home / "skills" / "alias"
        assert os.readlink(link) == "SKILL.md"
        assert link.read_text() == "internal skill"

    def test_absolute_link_keeps_its_target(self, tmp_path):
        source = tmp_path / "src"
        skills = source / "skills"
        skills.mkdir(parents=True)
        shared = tmp_path / "shared-skills"
        shared.mkdir()
        (shared / "SKILL.md").write_text("absolute skill")
        (skills / "shared").symlink_to(shared)

        result = _build(source, tmp_path / "temp")

        link = result.home / "skills" / "shared"
        assert os.readlink(link) == str(shared)
        assert link.exists()

    def test_a_link_that_already_dangled_does_not_abort_the_build(self, tmp_path):
        """Rebasing must not turn a source-side dangling link into a failure."""

        source = tmp_path / "src"
        skills = source / "skills"
        skills.mkdir(parents=True)
        (skills / "gone").symlink_to("nested/missing")

        result = _build(source, tmp_path / "temp")

        link = result.home / "skills" / "gone"
        assert result.home.is_dir()
        assert link.is_symlink()
        assert not link.exists()
