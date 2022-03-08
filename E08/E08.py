# David Chalifoux & Connor White
from math import exp

# Training data
# [input1, input1, output1, output2]
trainingData = [
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0]
]

# Input nodes
i1Node = {
    "value": 1,
    "weight_h1": .2,
    "weight_h2": -.3
}
i2Node = {
    "value": 1,
    "weight_h1": .4,
    "weight_h2": .3
}

# Hidden nodes
h1Node = {
    "value": .1,
    "weight_o1": .3,
    "weight_o2": -.2
}
h2Node = {
    "value": -.1,
    "weight_o1": .5,
    "weight_o2": -.4
}

# Output nodes
o1Node = {
    "value": -.2,
    "true": 0
}
o2Node = {
    "value": .3,
    "true": 1
}


def feedForward():
    # Calc h1
    temp = i1Node["weight_h1"] + i2Node["weight_h1"] + h1Node["value"]
    h1Node["feedForward"] = round(1 / (1+exp(temp * -1)), 2)

    # Calc h2
    temp = i1Node["weight_h2"] + i2Node["weight_h2"] + h2Node["value"]
    h2Node["feedForward"] = round(1 / (1+exp(temp * -1)), 2)

    # Calc o1
    temp = (h1Node["weight_o1"] * h1Node["feedForward"]) + \
        (h2Node["weight_o1"] * h2Node["feedForward"]) + o1Node["value"]
    o1Node["feedForward"] = round(1 / (1+exp(temp * -1)), 2)
    # Calc o2
    temp = (h1Node["weight_o2"] * h1Node["feedForward"]) + \
        (h2Node["weight_o2"] * h2Node["feedForward"]) + o2Node["value"]
    o2Node["feedForward"] = round(1 / (1+exp(temp * -1)), 2)


def calcErr():
    # Calc err of o1
    temp = o1Node["feedForward"] * \
        (1-o1Node["feedForward"])*(o1Node["true"]-o1Node["feedForward"])
    o1Node["error"] = round(temp, 2)
    # Calc err of o2
    temp = o2Node["feedForward"] * \
        (1-o2Node["feedForward"])*(o2Node["true"]-o2Node["feedForward"])
    o2Node["error"] = round(temp, 2)
    # Calc err of h1
    temp = h1Node["feedForward"] * (1 - h1Node["feedForward"]) * (
        (o1Node["error"] * h1Node["weight_o1"]) + (o2Node["error"] * h1Node["weight_o2"]))
    h1Node["error"] = round(temp, 3)

    # Calc err of h2
    temp = h2Node["feedForward"] * (1 - h2Node["feedForward"]) * (
        (o1Node["error"] * h2Node["weight_o1"]) + (o2Node["error"] * h2Node["weight_o2"]))
    h2Node["error"] = round(temp, 3)


def backPropagation():
    # calc weight and Bias Adjustment of o1
    tempNode = o1Node["value"] + o1Node["error"]
    print(tempNode)
    o1ToH1 = h1Node["weight_o1"] + o1Node["error"] * h1Node["feedForward"]
    o1ToH2 = h2Node["weight_o1"] + o1Node["error"] * h2Node["feedForward"]

    # calc weight and Bias Adjustment of o2
    tempNode = o2Node["value"] + o2Node["error"]
    print(tempNode)
    o2ToH1 = h1Node["weight_o1"] + o2Node["error"] * h1Node["feedForward"]
    o2ToH2 = h2Node["weight_o1"] + o2Node["error"] * h2Node["feedForward"]

    # calc weight and Bias Adjustment of h1
    tempNode = h1Node["value"] + h1Node["error"]
    print(tempNode)
    h1ToI1 = i1Node["weight_h1"] + h1Node["error"] * 1
    h1ToI2 = i2Node["weight_h1"] + h1Node["error"] * 1
    # calc weight and Bias Adjustment of h2
    tempNode = h2Node["value"] + h2Node["error"]
    print(tempNode)
    h2ToI1 = i1Node["weight_h2"] + h2Node["error"] * 1
    h2ToI2 = i2Node["weight_h2"] + h2Node["error"] * 1


feedForward()
calcErr()
backPropagation()