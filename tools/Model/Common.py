import numpy
import random
import tabulate
import typing
import torch
from collections import defaultdict

class Color:
    RESET    = "\033[0m"

    BLACK    = "\033[30m"
    RED      = "\033[31m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    MAGENTA  = "\033[35m"
    CYAN     = "\033[36m"
    WHITE    = "\033[37m"

    LBLACK   = "\033[90m"
    LRED     = "\033[91m"
    LGREEN   = "\033[92m"
    LYELLOW  = "\033[93m"
    LBLUE    = "\033[94m"
    LMAGENTA = "\033[95m"
    LCYAN    = "\033[96m"
    LWHITE   = "\033[97m"

def ParameterCount(model: torch.nn.Module) -> typing.DefaultDict[str, tuple]:
    r = defaultdict(int)
    for name, prm in model.named_parameters():
        size = prm.numel()
        frozen = not prm.requires_grad
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            if prefix in r:
                r[prefix] = (size + r[prefix][0], frozen and r[prefix][1])
            else:
                r[prefix] = (size, frozen)
    return r

def ParameterCountTable(model: torch.nn.Module, max_depth: int = 3) -> str:
    count: typing.DefaultDict[str, tuple] = ParameterCount(model)
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }

    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    table: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        if x >= 1e9:
            return "{:.3f}G".format(x / 1e9)
        if x >= 1e6:
            return "{:.3f}M".format(x / 1e6)
        if x >= 1e3:
            return "{:.3f}K".format(x / 1e3)
        return str(x)

    def color_print(x: str, frozen: bool):
        if frozen:
            return f"{Color.CYAN}{x}{Color.RESET}"
        else:
            return x

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, val in count.items():
            frozen = val[1]
            if name.count(".") == lvl and name.startswith(prefix):
                indent = " " * (lvl + 1)
                if name in param_shape:
                    table.append((indent + color_print(name, frozen), indent + color_print(str(param_shape[name]), frozen)))
                else:
                    table.append((indent + color_print(name, frozen), indent + color_print(format_size(val[0]), frozen)))
                    fill(lvl + 1, name + ".")

    table.append(("model", format_size(count.pop("")[0])))
    fill(0, "")

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(
        table, headers=["name", "#elements or shape"], tablefmt="pipe"
    )
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab

def RandomSeed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
