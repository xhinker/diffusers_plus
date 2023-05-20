#%% load up package
import sys
sys.path.append('../src/')

#%%
from diffusers_plus.tools.prompt_tools import AZ_SD_Prompt as asp

prompt = "a photo of [gates:musk:0.5]"
r = asp.parse_scheduled_prompts(text=prompt)
#r = asp.func_test(prompt)
print(r)



#%% lark test
import lark
schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
alternate: "[" prompt ("|" prompt)+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

prompt = "a photo of [gates:musk: 0.5]"
tree = schedule_parser.parse(prompt)
print(tree)

#%% tet collect_steps
def collect_steps(steps, tree):
    l = [steps]
    class CollectSteps(lark.Visitor):
        def scheduled(self, tree):
            tree.children[-1] = float(tree.children[-1])
            if tree.children[-1] < 1:
                tree.children[-1] *= steps
            tree.children[-1] = min(steps, int(tree.children[-1]))
            l.append(tree.children[-1])
        def alternate(self, tree):
            l.extend(range(1, steps+1))
    CollectSteps().visit(tree)
    return sorted(set(l))

r = collect_steps(10,tree)
print(r)
print(tree)

#%%
def at_step(step, tree):
    class AtStep(lark.Transformer):
        def scheduled(self, args):
            before, after, _, when = args
            yield before or () if step <= when else after
        def alternate(self, args):
            yield next(args[(step - 1)%len(args)])
        def start(self, args):
            def flatten(x):
                if type(x) == str:
                    yield x
                else:
                    for gen in x:
                        yield from flatten(gen)
            return ''.join(flatten(args))
        def plain(self, args):
            yield args[0].value
        def __default__(self, data, children, meta):
            for child in children:
                yield child
    return AtStep().transform(tree)

r = at_step(step=5,tree=tree)
print(r)




#%%
Tree('start', [Tree('prompt', [Tree('plain', [Token('__ANON_1', 'a photo of ')]), Tree('scheduled', [Tree('prompt', [Tree('plain', [Token('__ANON_1', 'gates')])]), Tree('prompt', [Tree('plain', [Token('__ANON_1', 'musk')])]), Token('NUMBER', '0.5')])])])

Tree('start', [Tree('prompt', [Tree('plain', [Token('__ANON_1', 'a photo of ')]), Tree('scheduled', [Tree('prompt', [Tree('plain', [Token('__ANON_1', 'gates')])]), Tree('prompt', [Tree('plain', [Token('__ANON_1', 'musk')])]), Token('WHITESPACE', ' '), Token('NUMBER', '0.5')])])])