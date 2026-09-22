import pandas as pd


data = {
    "Roll no": [1, 2, 3, 4, 5],
    "Name": ["Anu", "Ravi", "Meera", "Arjun", "Divya"],
    "Department": ["CSE", "ECE", "CSE", "ME", "CSE"],
    "Marks": [85, 72, 90, 68, 78],
}
df = pd.DataFrame(data)
print("Dataframe:")
print(df.to_string(index=False))

print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))
print("\nLast two rows:")
print(df.tail(2).to_string(index=False))
print("\nName and Marks columns:")
print(df[["Name", "Marks"]].to_string(index=False))
print("\nMarks more than 80:")
print(df[df["Marks"] > 80].to_string(index=False))
print("\nCSE department:")
print(df[df["Department"] == "CSE"].to_string(index=False))
print("\nHighest =", df["Marks"].max(), " Lowest =", df["Marks"].min(),
      " Avg =", round(df["Marks"].mean(), 2), " Total =", df["Marks"].sum())
print("\nAverage mark department wise:")
print(df.groupby("Department")["Marks"].mean().round(2).to_string())

df.to_csv("exp3_students.csv", index=False)
print("\nSaved to exp3_students.csv")
