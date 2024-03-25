import csv


def read_csv_to_string_array(file_path):
    string_array = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            string_array.append(row)
    return string_array


# Replace 'your_file.csv' with the path to your CSV file
file_path = 'archive/atis_intents_train.csv'
string_array = read_csv_to_string_array(file_path)
# print(string_array[0][1])
unique_list = []
for s in string_array:
    if s[0] not in unique_list:
        unique_list.append(s[0])
print(unique_list)
print(len(unique_list))
