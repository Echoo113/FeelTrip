from datasets import load_dataset
dataset = load_dataset("dair-ai/emotion")
print(dataset['train'][0])
#print entire dataset
for i in range(len(dataset['train'])):
    print(dataset['train'][i])
print("Number of samples in the train set:", len(dataset['train']))
print("Number of samples in the test set:", len(dataset['test']))
print("Number of samples in the validation set:", len(dataset['validation']))
print("Number of samples in the entire dataset:", len(dataset))