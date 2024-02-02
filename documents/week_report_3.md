# Week report, week 3

During this week the testing was initiated. Together with the course teacher it was agreed to focus the testing efforts on the actual neural network i.e. network.py. It was further agreed that the testing would consist of i) unit testing and ii) and a sanity check. The unit testing of the neural network is on a good track with 73% coverage. The sanity comparsion is in an ideation/planning phase, but the current idea is to:

* Simplify the input e.g. to 10pixels
* Set up a simplified network e.g. ready library
* Train the simplified network and own network with the simplified input
* Compare intermediate calculation steps between the simplified network and own network e.g. feedforward outputs.

Next week I would like to test the sanity comparsion idea in practice. At the moment it feels not too solid, so any guidance for a smoother start would be appreciated e.g. which pixels out of the original MNIST 28 x 28 pixels (784) would make most sense to start of with to simplify the data? Could it be 3 x 3 shaped square in the center of the images? Also first ideas where to look for suitable simplified networks for this purpouse would be helpful.

## Used hours

| Day   | Used time | Description                  |
| ----- | --------- | ---------------------------- |
| 30.1  | 1h        | Familiarizing with testing material      |
| 31.1  | 2h        | Setting up unit tests. Planning meeting aobut project testing              |
| 1.2  | 4h        | Unit testing |
| 2.2  | 2h        | Week report and initiating testing documentation |
| Total | 8h        |                              |
