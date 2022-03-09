# David Chalifoux & Connor White
from math import exp
import random

# Input nodes
i1Node = {
    "value": 1,
    "weight_h1": random.uniform(-.5, .5),
    "weight_h2": random.uniform(-.5, .5)
}
i2Node = {
    "value": 1,
    "weight_h1": random.uniform(-.5, .5),
    "weight_h2": random.uniform(-.5, .5)
}

# Hidden nodes
h1Node = {
    "value": random.uniform(-.5, .5),
    "weight_o1": random.uniform(-.5, .5),
    "weight_o2": random.uniform(-.5, .5)
}
h2Node = {
    "value": random.uniform(-.5, .5),
    "weight_o1": random.uniform(-.5, .5),
    "weight_o2": random.uniform(-.5, .5)
}

# Output nodes
o1Node = {
    "value": random.uniform(-.5, .5),
    "true": 0
}
o2Node = {
    "value": random.uniform(-.5, .5),
    "true": 1
}


def feedForward():
    # Calc h1
    temp = i1Node["weight_h1"] + i2Node["weight_h1"] + h1Node["value"]
    h1Node["feedForward"] = 1 / (1+exp(temp * -1))

    # Calc h2
    temp = i1Node["weight_h2"] + i2Node["weight_h2"] + h2Node["value"]
    h2Node["feedForward"] = 1 / (1+exp(temp * -1))

    # Calc o1
    temp = (h1Node["weight_o1"] * h1Node["feedForward"]) + \
        (h2Node["weight_o1"] * h2Node["feedForward"]) + o1Node["value"]
    o1Node["feedForward"] = 1 / (1+exp(temp * -1))
    # Calc o2
    temp = (h1Node["weight_o2"] * h1Node["feedForward"]) + \
        (h2Node["weight_o2"] * h2Node["feedForward"]) + o2Node["value"]
    o2Node["feedForward"] = 1 / (1+exp(temp * -1))


def calcErr():
    # Calc err of o1
    temp = o1Node["feedForward"] * \
        (1-o1Node["feedForward"])*(o1Node["true"]-o1Node["feedForward"])
    o1Node["error"] = temp
    # Calc err of o2
    temp = o2Node["feedForward"] * \
        (1-o2Node["feedForward"])*(o2Node["true"]-o2Node["feedForward"])
    o2Node["error"] = temp
    # Calc err of h1
    temp = h1Node["feedForward"] * (1 - h1Node["feedForward"]) * (
        (o1Node["error"] * h1Node["weight_o1"]) + (o2Node["error"] * h1Node["weight_o2"]))
    h1Node["error"] = temp
    # Calc err of h2
    temp = h2Node["feedForward"] * (1 - h2Node["feedForward"]) * (
        (o1Node["error"] * h2Node["weight_o1"]) + (o2Node["error"] * h2Node["weight_o2"]))
    h2Node["error"] = temp


def backPropagation():
    # Calc weight and bias adjustment of o1
    o1Node["value"] = o1Node["value"] + o1Node["error"]
    # Calc weight and bias adjustment of o2
    o2Node["value"] = o2Node["value"] + o2Node["error"]
    # Calc weight and bias adjustment of h1
    h1Node["weight_o1"] = h1Node["weight_o1"] + \
        o1Node["error"] * h1Node["feedForward"]
    h1Node["weight_o2"] = h1Node["weight_o2"] + \
        o2Node["error"] * h1Node["feedForward"]
    h1Node["value"] = h1Node["value"] + h1Node["error"]
    # Calc weight and bias adjustment of h2
    h2Node["weight_o1"] = h2Node["weight_o1"] + \
        o1Node["error"] * h2Node["feedForward"]
    h2Node["weight_o2"] = h2Node["weight_o2"] + \
        o2Node["error"] * h2Node["feedForward"]
    h2Node["value"] = h2Node["value"] + h2Node["error"]
    # Calc weight and bias adjustment of i1
    i1Node["weight_h1"] = i1Node["weight_h1"] + \
        h1Node["error"] * i1Node["value"]
    i1Node["weight_h2"] = i1Node["weight_h2"] + \
        h2Node["error"] * i1Node["value"]
    # Calc weight and bias adjustment of i2
    i2Node["weight_h1"] = i2Node["weight_h1"] + \
        h1Node["error"] * i2Node["value"]
    i2Node["weight_h2"] = i2Node["weight_h2"] + \
        h2Node["error"] * i2Node["value"]


def train(trainingData):
    for trainingSet in trainingData:
        print("Training on", trainingSet)

        # Setup nodes
        i1Node["value"] = trainingSet[0]
        i2Node["value"] = trainingSet[1]
        o1Node["true"] = trainingSet[2]
        o2Node["true"] = trainingSet[3]

        # Run full back-propagation
        for i in range(100):
            feedForward()
            calcErr()
            backPropagation()

        # Determine result
        print(o1Node)
        print(o2Node)
        print("")


# Training data
# [input1, input1, output1, output2]
trainingData = [
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0]
]
train(trainingData)
