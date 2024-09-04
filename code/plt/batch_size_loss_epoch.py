import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
# check if I'm in the right directory
if not os.getcwd().endswith('graduate'):
    os.chdir(os.getenv('PROJECT_PATH'))

markers = ['o', 's', '<', '^', 'v']
start = 16
for i in range(5):
    # Load the data from the CSV file
    csv = f'server/SAG_{start}_bs/SAG_train.csv'
    data = pd.read_csv(csv)

    # Plot the data
    plt.plot(data['epoch'], data['recall'], marker=markers[i], linewidth=1, markersize=3, markevery=2, label=f'{start}')

    start = start * 2

# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend(loc='upper right')

# Save the figure
plt.savefig('paper/figures/batch_size_recall_epoch.pdf')

# Display the plot
plt.show()