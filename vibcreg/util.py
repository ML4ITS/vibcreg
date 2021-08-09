
from pathlib import Path

import git


def get_git_root() -> Path:
        git_repo = git.Repo(__file__, search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        return Path(git_root)


if __name__ == "__main__":
        this_path = Path(__file__)
        git_root = get_git_root()
        print(git_root, type(git_root))