# David Chalifoux & Connor White
from math import exp
import random

# Input nodes
i1Node = {"value": 1, "weight_h1": 0.2, "weight_h2": -0.3}
i2Node = {"value": 1, "weight_h1": 0.4, "weight_h2": 0.3}

# Hidden nodes
h1Node = {"value": 0.1, "weight_o1": 0.3, "weight_o2": -0.2}
h2Node = {"value": -0.1, "weight_o1": 0.5, "weight_o2": -0.4}

# Output nodes
o1Node = {"value": -0.2, "true": 0}
o2Node = {"value": 0.3, "true": 1}


def feedForward():
    # Calc h1
    temp = i1Node["weight_h1"] + i2Node["weight_h1"] + h1Node["value"]
    h1Node["feedForward"] = 1 / (1 + exp(temp * -1))

    # Calc h2
    temp = i1Node["weight_h2"] + i2Node["weight_h2"] + h2Node["value"]
    h2Node["feedForward"] = 1 / (1 + exp(temp * -1))

    # Calc o1
    temp = (
        h1Node["weight_o1"] * h1Node["feedForward"]
        + h2Node["weight_o1"] * h2Node["feedForward"]
        + o1Node["value"]
    )
    o1Node["feedForward"] = 1 / (1 + exp(temp * -1))

    # Calc o2
    temp = (
        h1Node["weight_o2"] * h1Node["feedForward"]
        + h2Node["weight_o2"] * h2Node["feedForward"]
        + o2Node["value"]
    )
    o2Node["feedForward"] = 1 / (1 + exp(temp * -1))


def calcErr(learningRate=1):
    # Calc err of o1
    temp = (
        o1Node["feedForward"]
        * (1 - o1Node["feedForward"])
        * (o1Node["true"] - o1Node["feedForward"])
    )
    o1Node["error"] = learningRate * temp
    # Calc err of o2
    temp = (
        o2Node["feedForward"]
        * (1 - o2Node["feedForward"])
        * (o2Node["true"] - o2Node["feedForward"])
    )
    o2Node["error"] = learningRate * temp
    # Calc err of h1
    temp = (
        h1Node["feedForward"]
        * (1 - h1Node["feedForward"])
        * (
            (o1Node["error"] * h1Node["weight_o1"])
            + (o2Node["error"] * h1Node["weight_o2"])
        )
    )
    h1Node["error"] = learningRate * temp
    # Calc err of h2
    temp = (
        h2Node["feedForward"]
        * (1 - h2Node["feedForward"])
        * (
            (o1Node["error"] * h2Node["weight_o1"])
            + (o2Node["error"] * h2Node["weight_o2"])
        )
    )
    h2Node["error"] = learningRate * temp


def backPropagation():
    # Calc weight and bias adjustment of o1
    o1Node["value"] = o1Node["value"] + o1Node["error"]
    # Calc weight and bias adjustment of o2
    o2Node["value"] = o2Node["value"] + o2Node["error"]
    # Calc weight and bias adjustment of h1
    h1Node["weight_o1"] = h1Node["weight_o1"] + o1Node["error"] * h1Node["feedForward"]
    h1Node["weight_o2"] = h1Node["weight_o2"] + o2Node["error"] * h1Node["feedForward"]
    h1Node["value"] = h1Node["value"] + h1Node["error"]
    # Calc weight and bias adjustment of h2
    h2Node["weight_o1"] = h2Node["weight_o1"] + o1Node["error"] * h2Node["feedForward"]
    h2Node["weight_o2"] = h2Node["weight_o2"] + o2Node["error"] * h2Node["feedForward"]
    h2Node["value"] = h2Node["value"] + h2Node["error"]
    # Calc weight and bias adjustment of i1
    i1Node["weight_h1"] = i1Node["weight_h1"] + h1Node["error"] * i1Node["value"]
    i1Node["weight_h2"] = i1Node["weight_h2"] + h2Node["error"] * i1Node["value"]
    # Calc weight and bias adjustment of i2
    i2Node["weight_h1"] = i2Node["weight_h1"] + h1Node["error"] * i2Node["value"]
    i2Node["weight_h2"] = i2Node["weight_h2"] + h2Node["error"] * i2Node["value"]


def train(trainingData, times, learningRate=1):
    print("Starting training...\n")
    for epoch in range(times):
        for trainingSet in trainingData:
            # Setup nodes
            i1Node["value"] = trainingSet[0]
            i2Node["value"] = trainingSet[1]
            o1Node["true"] = trainingSet[2]
            o2Node["true"] = trainingSet[3]

            # Run full back-propagation
            feedForward()
            calcErr(learningRate=learningRate)
            backPropagation()
        print(epoch + 1, "epochs completed", end="\r")

    print("\nTraining finished ✅")


def test(testingData):
    print("\nStarting testing...\n")
    for testingSet in testingData:
        # Setup nodes
        i1Node["value"] = testingSet[0]
        i2Node["value"] = testingSet[1]
        o1Node["true"] = testingSet[2]
        o2Node["true"] = testingSet[3]

        # Calc values
        feedForward()

        print("O1:", o1Node)
        print("O2:", o2Node)
        print()
        print("H1:", h1Node)
        print("H2:", h2Node)
        print()
        print("I1:", i1Node)
        print("I2:", i2Node)
        print()

        # Validate
        print("Set:", testingSet)
        output1Estimate = round(o1Node["feedForward"])
        output2Estimate = round(o2Node["feedForward"])
        output1True = o1Node["true"]
        output2True = o2Node["true"]
        print("Estimated output 1:", output1Estimate)
        print("True value output 1:", output1True)
        if output1Estimate == output1True:
            print("Success ✅")
        else:
            print("Fail ❌")
        print("Estimated output 2:", output2Estimate)
        print("True value output 2:", output2True)
        if output2Estimate == output2True:
            print("Success ✅")
        else:
            print("Fail ❌")
        print()


# Training data
# [input1, input1, output1, output2]
trainingData = [
    # [1, 1, 0, 1],
    # [1, 0, 1, 0],
    # [0, 1, 1, 0],
    [0, 0, 0, 1],
]
train(trainingData, times=10000, learningRate=1)
test(trainingData)
