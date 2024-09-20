from dataclasses import dataclass

import tyro


@dataclass
class test_dataclass:
    a: int = 0


if __name__ == "__main__":
    args = tyro.cli(test_dataclass)
    print(args)
