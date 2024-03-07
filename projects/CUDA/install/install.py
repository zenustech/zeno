import os
import shutil
import os.path as osp
from typing import Any
from argparse import ArgumentParser
from git import Repo
from icecream import ic

parser = ArgumentParser()
parser.add_argument("--zeno_bin_dir", type=str)
parser.add_argument("--zeno_pyzpc_repo_dir", type=str)
parser.add_argument("--git_proxy", type=str, default="")

PYZPC_GIT_URL = "https://github.com/zenustech/POC_zpc_jit.git"


def pull_git_module_recursive(module: Repo) -> None:
    for submodule in module.submodules:
        submodule.update(init=True)
        ic(submodule.name, submodule.abspath)
        if submodule.name.endswith("tpls"):
            submodule.module().git.checkout("main")
        else:
            submodule.module().git.checkout("master")
        os.system(f"cd {submodule.abspath} && git pull")
        pull_git_module_recursive(submodule.module())


def main(args: Any) -> None:
    print("install pyzpc!!!!")
    git_proxy: str = args.git_proxy
    repo_dir: str = args.zeno_pyzpc_repo_dir
    download_dir: str = osp.join(
        osp.dirname(args.zeno_pyzpc_repo_dir), "download"
    )
    if not osp.exists(repo_dir):
        os.makedirs(repo_dir)
    if osp.exists(download_dir):
        shutil.rmtree(download_dir)
    use_proxy: bool = (len(git_proxy) != 0) and (git_proxy != "none")
    if use_proxy:
        os.environ["all_proxy"] = git_proxy
    repo: Repo | None = None
    if use_proxy:
        repo = Repo.clone_from(
            PYZPC_GIT_URL,
            download_dir,
            config=f"https.proxy='{git_proxy}'",
            allow_unsafe_options=True,
        )
    else:
        repo = Repo.clone_from(
            PYZPC_GIT_URL,
            download_dir,
            allow_unsafe_options=True,
            recurse_submodules=True,
        )
    assert repo is not None
    pull_git_module_recursive(repo)

    zeno_bin_dir: str = args.zeno_bin_dir
    if not osp.exists(zeno_bin_dir):
        raise FileNotFoundError(f"ZENO binary folder {zeno_bin_dir} not found.")

    shutil.copytree(
        download_dir,
        repo_dir,
        copy_function=shutil.copyfile,
        dirs_exist_ok=True,
    )
    os.system(f"{zeno_bin_dir}/bin/python -m pip install {repo_dir}")
    os.system(f"{zeno_bin_dir}/bin/python -m pip install numpy")
    os.system(f"{zeno_bin_dir}/bin/python -m pip install icecream")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
