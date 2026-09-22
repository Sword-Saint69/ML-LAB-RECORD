import numpy as np


marks = np.array([
    [78, 85, 92, 88, 76],
    [65, 72, 80, 69, 74],
    [90, 88, 95, 91, 89],
])
print("Marks (3 students x 5 subjects):")
print(marks)

total = marks.sum()
average = marks.mean()
highest = marks.max()
pos = np.unravel_index(marks.argmax(), marks.shape)
print("Total =", total)
print("Average =", round(float(average), 2))
print("Highest score =", highest, "by student", pos[0] + 1,
      "in subject", pos[1] + 1)

stud_total = marks.sum(axis=1)
print("Student-wise total =", stud_total)
sub_avg = marks.mean(axis=0)
print("Subject-wise average =", np.round(sub_avg, 2))
