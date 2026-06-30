"""generate python interface file for pybind_supa library.
the pyi file helps codeing test.
"""

import sys
import pydoc
import inspect
import re
import os

if not os.path.exists("/sys/class/biren/card_0"):
    print("exit due to no card found.")
    sys.exit(0)

import pybind_supa

args_pattern = re.compile(r"<([^:]*): \d*>")
"""patter to remove enum value '<...>' out of args.
'layout: BRLayoutEnum = <BRLayoutEnum.BUFFER_ANY: 1>' -> 'layout: BRLayoutEnum = BRLayoutEnum.BUFFER_ANY'
"""


class HintDoc(pydoc.Doc):
    def __init__(self) -> None:
        super().__init__()
        self.indent = 0

    def format(self, line: str, is_comment: bool = False):
        cap = '"""' if is_comment else ""
        return f"{'    ' * self.indent}{cap}{line}{cap}"

    def docmodule(self, object, name=None, mod=None) -> str:
        result = []
        push = result.append
        desc = pydoc.getdoc(object)
        if desc:
            push(self.format(desc, True))
        push("from typing import List, Optional")
        classes = inspect.getmembers(object, pydoc._isclass)
        funcs = inspect.getmembers(object, inspect.isroutine)

        if classes:
            for key, value in classes:
                push(self.document(value, key, name))

        if funcs:
            for key, value in funcs:
                push(self.document(value, key, name))

        return "\n".join(result)

    def docclass(self, object, name=None, mod=None) -> str:
        result = []
        push = result.append
        push(self.format(f"class {name}:"))
        self.indent += 1
        doc = pydoc.getdoc(object)
        if doc:
            push(self.format(doc, True))

        for name, kind, cls, value in inspect.classify_class_attrs(object):
            if name.startswith("__"):
                continue
            if kind == "data":
                push(self.docdata(value, name))
            elif kind == "property":
                push(self.docproperty(value, name))
            elif "method" in kind:
                push(self.docroutine(value, name, kind))
            else:
                print(kind, name)
                # push(self.document(value, name))

        self.indent -= 1
        return "\n".join(result)

    @staticmethod
    def strip_classname(src: str) -> str:
        return src.replace("pybind_supa.", "").replace("Exception.", "").replace(" --", "")

    @staticmethod
    def strip_enum(src: str) -> str:
        global args_pattern
        if args_pattern is None:
            args_pattern = re.compile(r"<([^:]*): \d*>")
            """patter to remove enum value '<...>' out of args.
            'layout: BRLayoutEnum = <BRLayoutEnum.BUFFER_ANY: 1>' -> 'layout: BRLayoutEnum = BRLayoutEnum.BUFFER_ANY'
            """
        return args_pattern.sub(r"\1", src)

    def docroutine(self, object, name=None, mod=None) -> str:
        result = []
        if mod == "static method":
            result.append(self.format("@staticmethod"))
            object = object.__get__(self)
        doc = pydoc.getdoc(object)
        if not doc:
            signature, doc = f"{name}(...)", ""
        else:
            parts = doc.split("\n", 1)
            if len(parts) == 1:
                signature, doc = parts[0], ""
            else:
                signature, doc = parts
            signature = HintDoc.strip_classname(HintDoc.strip_enum(signature))
        result.append(self.format(f"def {signature}:"))
        self.indent += 1
        if doc:
            doc = doc.strip()
            result.append(self.format(f'"""{doc}"""'))
        else:
            result.append(self.format("..."))
        self.indent -= 1

        return "\n".join(result)

    def docother(self, object, name=None, mod=None) -> str:
        return f'"""other {name}"""'

    def docproperty(self, object, name=None, mod=None) -> str:
        result = []
        result.append(self.format("@property"))
        doc = pydoc.getdoc(object)
        if object.fget.__doc__:
            ret_type = object.fget.__doc__.split("->")[1].strip()
            ret_type = HintDoc.strip_classname(ret_type)
            signature = f"{name}(self) -> {ret_type}"
        else:
            signature = f"{name}(self)"
        if doc:
            doc = f'"""{doc}"""'
        else:
            doc = "..."

        result.append(self.format(f"def {signature}:"))
        self.indent += 1
        result.append(self.format(f"{doc}"))
        self.indent -= 1
        return "\n".join(result)

    def docdata(self, object, name=None, mod=None) -> str:
        return self.format(f"{name} = {object.value}" if hasattr(object, "value") else f"{name} : object")


if __name__ == "__main__":
    dst_path = sys.argv[1]

    all_doc = pydoc.render_doc(pybind_supa, title=f'"%s"', renderer=HintDoc())
    with open(f"{dst_path}/pybind_supa.pyi", "w") as fo:
        fo.write(all_doc)
