#import the necessary packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "/Users/ulises.rey/Desktop/time_spent.csv"
df = pd.read_csv(path)
print(df.head(10))

#drop some rows based if the student name is on the list students_to_drop
students_to_drop = ["Lars", "Nisa", "Daniel M", "Sonja", "Barbara", "Giulietta",  "Katarina", "Maximilian", "Tanja", "Nadja" ]
df = df[~df.Student.isin(students_to_drop)]


# Generate a violin plot of the column 'Time Hrs' in the dataframe df based on the variable Student, sorted by mean of the Time Hrs column
sns.violinplot(x="Student", y="Time (Hrs)", data=df, order=df.groupby('Student')['Time (Hrs)'].mean().sort_values().index)


# Sort the x labels based on the mean of the column 'Time (Hrs)'


#draw a horizontal line at the mean of the df valy of the column 'Time (Hrs)'
plt.axhline(df['Time (Hrs)'].mean(), color='red', linestyle='dashed', linewidth=2)
#rotate the x-axis labels
plt.xticks(rotation=90)
plt.show()