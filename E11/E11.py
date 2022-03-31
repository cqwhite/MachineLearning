# Connor White & David Chalifoux
# Derived from: https://janakiev.com/blog/pytorch-iris/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

iris = load_iris()
X = iris["data"]
y = iris["target"]

# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training, testing, and validation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4
)  # Split for training
X_test, X_validate, y_test, y_validate = train_test_split(
    X_test, y_test, test_size=0.5
)  # Split for validation

print("Data length:", len(X_scaled))
print("Training data length:", len(X_train))
print("Testing data length:", len(X_test))
print("Validation data length:", len(X_validate))


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


model = Model(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 200
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()
X_validate = Variable(torch.from_numpy(X_validate)).float()
y_validate = Variable(torch.from_numpy(y_validate)).long()

for epoch in range(EPOCHS):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        # Test
        y_pred_test = model(X_test)
        correct_test = (torch.argmax(y_pred_test, dim=1) == y_test).type(
            torch.FloatTensor
        )

        # Validate
        y_pred_validate = model(X_validate)
        correct_validate = (torch.argmax(y_pred_validate, dim=1) == y_validate).type(
            torch.FloatTensor
        )
        print(
            "Training epoch",
            epoch + 1,
            "loss %.4f" % loss.item(),
            "test accuracy %.4f" % correct_test.mean().item(),
            "validation accuracy %.4f" % correct_validate.mean().item(),
        )
