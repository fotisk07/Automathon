import numpy as np
import torch
import torch.nn as nn
import tqdm
import segmentation_models_pytorch as smp


BATCH_SIZE = 16

# Load data
dataset_train = np.load('dataset_train.npy')
dataset_train_label = np.load('dataset_train_label.npy')
dataset_test = np.load('dataset_test.npy')

# Reshape data (N, 28, 28, 1) -> (n, BATCH_SIZE, 28, 28, 1)
dataset_train = dataset_train.reshape((-1, BATCH_SIZE, 28, 28, 3))
dataset_train_label = dataset_train_label.reshape((-1, BATCH_SIZE, 2))
dataset_test = dataset_test.reshape((-1, BATCH_SIZE, 28, 28, 1))

x_train = dataset_train[:, :, :, :, :1]
y1_train = dataset_train[:, :, :, :, 1:2]
y2_train = dataset_train[:, :, :, :, 2:]
label_train = dataset_train_label[:, :, 0]
x_test = dataset_test

# Convert to torch tensor
x_train = torch.from_numpy(x_train).float()
y1_train = torch.from_numpy(y1_train).float()
y2_train = torch.from_numpy(y2_train).float()
label_train = torch.from_numpy(label_train).float()
x_test = torch.from_numpy(x_test).float()


# Define model
class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same')
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding='same')

	def forward(self, x):
		"""

		:param x: torch.Tensor: (BATCH_SIZE, 28, 28, 1)
		:return: y1, y2: torch.Tensor: (BATCH_SIZE, 28, 28, 1)
		"""

		x = x.permute(0, 3, 1, 2)  # (BATCH_SIZE, 1, 28, 28)

		x = self.conv1(x)  # (BATCH_SIZE, 16, 28, 28)
		x = self.conv2(x)  # (BATCH_SIZE, 2, 28, 28)

		y1 = x[:, 0:1, :, :]  # (BATCH_SIZE, 1, 28, 28)
		y2 = x[:, 1:2, :, :]  # (BATCH_SIZE, 1, 28, 28)

		y1 = y1.permute(0, 2, 3, 1)  # (BATCH_SIZE, 28, 28, 1)
		y2 = y2.permute(0, 2, 3, 1)  # (BATCH_SIZE, 28, 28, 1)

		return y1, y2


model = Net()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train with progress bar
for epoch in range(5):
	pbar = tqdm.tqdm(range(len(x_train)))
	for i in pbar:
		x = x_train[i]
		y1 = y1_train[i]
		y2 = y2_train[i]

		y1_pred, y2_pred = model(x)

		loss = loss_fn(y1_pred, y1) + loss_fn(y2_pred, y2)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		pbar.set_description('Epoch: {}, Loss: {}'.format(epoch, loss.item()))


# Test
print("Testing...")
preds = []
for i in range(len(x_test)):
	x = x_test[i]
	y1_pred, y2_pred = model(x)

	pred = torch.cat([y1_pred, y2_pred], dim=3)
	pred = pred.detach().numpy()
	preds.append(pred)

preds = np.concatenate(preds, axis=0)
np.save('preds.npy', preds)
