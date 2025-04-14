from transformers import AutoTokenizer,BertTokenizer

# Load the dictionary and tokenizer
token = BertTokenizer.from_pretrained(r"/Users/shokhzodjuraev/Desktop/AI-Duke/AI-BasicStart/AI-Duke/ai-basic-day07/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
# print(token)

sents = ["价格在这个地段属于适中, 附近有早餐店,小饭店, 比较方便,无早也无所",
         "房间不错,只是上网速度慢得无法忍受,打开一个网页要等半小时,连邮件都无法收。另前台工作人员服务态度是很好，只是效率有得改善。"]

# Batch encode sentences
out = token.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0],sents[1]],
    add_special_tokens=True,
    # Truncate when the sentence length exceeds max_length
    truncation=True,
    max_length=50,
    # Pad with 0 to max_length
    padding="max_length",
    # Can take values tf, pt, np, default is list
    return_tensors=None,
    # Return attention_mask
    return_attention_mask=True,
    return_token_type_ids=True,
    return_special_tokens_mask=True,
    # Return length
    return_length=True
)
# input_ids are the encoded words
# token_type_ids: positions of the first sentence and special symbols are 0, positions of the second sentence are 1 (only for context encoding)
# special_tokens_mask: positions of special symbols are 1, other positions are 0
# print(out)
for k,v in out.items():
    print(k,";",v)

# Decode text data
print(token.decode(out["input_ids"][0]),token.decode(out["input_ids"][1]))


