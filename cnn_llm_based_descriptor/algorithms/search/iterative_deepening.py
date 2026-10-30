"""Iterative deepening: repeat depth-limited DFS with fresh visited states."""


def solve(context):
    for limit in range(len(context.targets) + 1):
        stack, seen = [context.initial], set()
        if limit:
            context.audit["generated_states"] += 1
        while stack:
            node = stack.pop()
            state = node[:2]
            if state in seen:
                continue
            seen.add(state)
            context.visit(node, len(stack), limit)
            if node[0] == context.goal:
                return context.result(node)
            if len(node[3]) < limit:
                for child in reversed(list(context.children(node))):
                    stack.append(child); context.pushed(stack)
    raise RuntimeError("No review plan found.")
