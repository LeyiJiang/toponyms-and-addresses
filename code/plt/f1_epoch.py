import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
# check if I'm in the right directory
if not os.getcwd().endswith('graduate'):
    os.chdir(os.getenv('PROJECT_PATH'))

# Load the data from the CSV file
data = pd.read_csv('server/SAG_150_e.csv')

# Plot the data
plt.plot(data['epoch'], data['recall'])

# Add labels to the axes
plt.xlabel('Epoch')
plt.ylabel('Recall')

# Save the figure
plt.savefig('paper/figures/recall_epoch.pdf')

# Display the plot
plt.show()