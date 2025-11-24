# Adding extinct classifier as guidance 
Does this help us discover template-free "extinct" peptides? 

## Results 
### See finished runs on wandb here: 

https://wandb.ai/nmaus-penn/diffbo-guide-extinct/workspace?nw=nwusernmauspenn

### Comparing versions 
1. ts
2. ddim_repaint
3. ddim_repaint + extinct guidance w/ scale = 1 
4. ddim_repaint + extinct guidance w/ scale = 4
5. ddim_repaint + extinct guidance w/ scale = 16
6. ddim_repaint + extinct guidance w/ scale = 64

### Note 
Wandb logs show ~same performance for each version (1-6) above in terms of final objective value, but the question is whether the versions that add "extinct guidance" successfully discover high-scoring peptides that are ALSO more likely to be extinct. If so, that might be interesting enough to write a Bio paper (we could send some sequences to Cesar's lab folks for lab validation). 

### Data from each run (top-k peptides and scores)
The top k solutions (peptides and scores) found by each run are here: 

[gdrive](https://drive.google.com/drive/folders/1ODKliJYnxLv3sbSi3k6kw-lInc3ROezJ?usp=sharing)

In folder: peptides-unconstrained-w-and-wout-extinct-guide

# TODO (Steps): 

1. Use wandb to see which run in google drive folder above corresponds to which version (1-6) above (data in google drive organized according to wandb run name for each run). 

2. Use the extinct classifier we trained to actually classify the top-k peptides found by each version (1-6). --> does extinct guidance lead to more high-scoring peptides that are classified as extinct? If so, send those sequences to Cesar's lab folks for validation! 

