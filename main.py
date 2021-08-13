import math
import operator
import random

from deap import base, creator, gp, tools, algorithms

tournament_size = 3
min_terminal = -5
max_terminal = 5
max_depth = 17
pop_size = 100
num_generations = 100


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def multiply_itself(x):
    return x * x


pset = gp.PrimitiveSetTyped("MAIN", [float], float)
pset.renameArguments(ARG0='x')

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(multiply_itself, [float], float)
# pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

for x in range(min_terminal, max_terminal):
    pset.addTerminal(x, float)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def fitness_evalaution(x):
    if x > 0:
        return (1 / x) + (math.sin(x))
    else:
        return (2 * x) + (x * x) + 3.0


def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    errors = list()
    for x in points:
        errors.append(multiply_itself(func(x) - fitness_evalaution(x)))

    return sum(errors) / len(errors),


toolbox.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)])
toolbox.register("select", tools.selTournament, tournsize=tournament_size)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))


def main():
    random.seed(318)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, num_generations,
                                   halloffame=hof, verbose=True)

    tree = gp.PrimitiveTree(hof[0])
    print(str(tree))
    function = toolbox.compile(expr=hof[0])
    return pop, log, hof


if __name__ == '__main__':
    main()
