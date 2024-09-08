#!/usr/bin/env python
# coding: utf-8

# In[14]:


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn


# In[5]:


file_path = "/kaggle/input/ferdousi-text/ferdousi.txt"
with open(file_path, "r", encoding="utf-8") as file:
    dataset = file.readlines()
sentences = [sentence.strip() for sentence in dataset]
sentences = sentences[2:]
input_sentences = sentences[:-1]
output_sentences = sentences[1:]


# In[6]:


input_train = input_sentences[:int(len(input_sentences)*0.9)]
input_test = input_sentences[int(len(input_sentences)*0.9):]
output_train = output_sentences[:int(len(output_sentences)*0.9)]
output_test = output_sentences[int(len(output_sentences)*0.9):]


# In[8]:


tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa")
model = AutoModelForCausalLM.from_pretrained("HooshvareLab/gpt2-fa")


# In[9]:


tokenizer.pad_token = tokenizer.eos_token
train_in_tokens = tokenizer(input_train, padding=True,return_tensors="pt")
test_in_tokens = tokenizer(input_test, padding=True,return_tensors="pt")
train_out_tokens = tokenizer(output_train, padding=True,return_tensors="pt")
test_out_tokens = tokenizer(output_test, padding=True,return_tensors="pt")


# In[15]:


#model = nn.DataParallel(model, device_ids=[0, 1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[16]:


train_in_ids = train_in_tokens.input_ids
train_out_ids = train_out_tokens.input_ids
test_in_ids = test_in_tokens.input_ids
test_out_ids = test_out_tokens.input_ids

train_mask = train_in_tokens.attention_mask
test_mask = test_in_tokens.attention_mask


# In[17]:


class Dataset(Dataset):
    def __init__(self, in_ids, mask, out_ids):
        self.input_ids = in_ids
        self.output_ids = out_ids
        self.attention_mask = mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        data = [self.input_ids[idx],self.attention_mask[idx],self.output_ids[idx]]
        return data


# In[18]:


train_dataset = Dataset(train_in_ids,train_mask,train_out_ids)
test_dataset = Dataset(test_in_ids,test_mask,test_out_ids)


# In[24]:


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)


# In[25]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()


# In[26]:


num_epochs=20
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    
    for batch in train_loader:
        in_ids,masks,out_ids = batch
        in_ids=in_ids.to(device)
        masks=masks.to(device)
        out_ids=out_ids.to(device)
        optimizer.zero_grad()
        output = model(input_ids=in_ids, attention_mask=masks).logits
        loss = criterion(output.view(-1, output.shape[-1]), out_ids.view(-1))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}")


# In[60]:


#model = nn.DataParallel(model, device_ids=[0, 1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[61]:


num_epochs=20
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    
    for batch in train_loader:
        in_ids,masks,out_ids = batch
        in_ids=in_ids.to(device)
        masks=masks.to(device)
        out_ids=out_ids.to(device)
        optimizer.zero_grad()
        output = model(input_ids=in_ids, attention_mask=masks).logits
        loss = criterion(output.view(-1, output.shape[-1]), out_ids.view(-1))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}")


# In[62]:


torch.save(model.state_dict(), "GP2_fine_tune.pth")


# In[191]:


def test_model(model,data_loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            in_ids,masks,out_ids = batch
            in_ids=in_ids.to(device)
            masks=masks.to(device)
            out_ids=out_ids.to(device)
            output = model(input_ids=in_ids, attention_mask=masks).logits
            loss = criterion(output.view(-1, output.shape[-1]), out_ids.view(-1))
            total_loss += loss.item()/256
    return total_loss


# In[192]:


evaluation_metric = test_model(model,test_loader)


# In[193]:


evaluation_metric 


# In[217]:


def Text_generator(input_sentence,model,tokenizer):
    input_ids = tokenizer.encode(input_sentence, add_special_tokens=True, return_tensors="pt").to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float().to(device)  # Creating attention mask
#model = model.module
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask,  min_length=2*len(input_sentence)-4, num_return_sequences=1)
    generated_text = output[0].tolist()
    generated_text = generated_text[len(input_ids[0]):]  
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated  = generated_text[len(input_sentence):len(generated_text)]
    print(input_sentence)
    print(generated)
    


# In[127]:


input_sentence = "رستم برفت تا کند جنگی سخت با سهراب"
Text_generator(input_sentence,model,tokenizer)


# In[130]:


input_sentence = "به خال هندویش بخشم سمرقند و بخارا را"
Text_generator(input_sentence,model,tokenizer)


# In[177]:


input_sentence = "در جایی که عقاب تیز پرکشد "
Text_generator(input_sentence,model,tokenizer)


# In[ ]:





# In[ ]:




