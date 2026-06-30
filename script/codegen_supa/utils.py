import os
from argparse import Namespace
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import Self
from pathlib import Path

from torchgen.code_template import CodeTemplate

from torchgen.utils import (
    assert_never,
    concatMap,
    context,
    FileManager,
    make_file_manager,
    mapMaybe,
    NamespaceHelper,
    Target,
    T,
    string_stable_hash,
)


def fm_write_sharded(
    self,
    filename: str,
    items: Iterable[T],
    *,
    key_fn: Callable[[T], str],
    env_callable: Callable[[T], Dict[str, List[str]]],
    num_shards: int,
    base_env: Optional[Dict[str, Any]] = None,
    sharded_keys: Set[str],
) -> None:
    """overwrite default FileManager.write_sharded."""
    shard: Dict[str, Any] = {"shard_id": "Everything"}

    if base_env is not None:
        shard.update(base_env)

    for key in sharded_keys:
        if key in shard:
            assert isinstance(shard[key], list), "sharded keys in base_env must be a list"
            shard[key] = shard[key].copy()
        else:
            shard[key] = []

    def merge_env(into: Dict[str, List[str]], from_: Dict[str, List[str]]) -> None:
        for k, v in from_.items():
            assert k in sharded_keys, f"undeclared sharded key {k}"
            into[k] += v

    if self.dry_run:
        # Dry runs don't write any templates, so incomplete environments are fine
        items = ()

    for item in items:
        env = env_callable(item)
        merge_env(shard, env)

    self.write_with_template(filename, filename, lambda: shard)


def fm_write_with_template(
    self,
    filename: str,
    template_fn: str,
    env_callable: Callable[[], Union[str, Dict[str, Any]]],
) -> None:
    filename = os.path.join(self.install_dir, filename)
    # assert filename not in self.filenames, "duplicate file write {filename}"
    # self.filenames.add(filename)
    print(f"Procesing {filename}")
    if not self.dry_run:
        substitute_out = self.substitute_with_template(
            template_fn=template_fn,
            env_callable=env_callable,
        )
        self._write_if_changed(filename=filename, contents=substitute_out)


FileManager.write_sharded = fm_write_sharded
FileManager.write_with_template = fm_write_with_template


class SwitchedTemplateFolder(object):
    """temporarily change filemanager's template dir."""

    def __init__(self, fm: FileManager, template_dir: str) -> None:
        self.fm: FileManager = fm
        self.dir: str = Path(template_dir) if isinstance(self.fm.template_dir, Path) else template_dir
        self.old_dir: str = None

    def __enter__(self):
        self.old_dir = self.fm.template_dir
        self.fm.template_dir = self.dir
        return self.fm

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        self.fm.template_dir = self.old_dir
