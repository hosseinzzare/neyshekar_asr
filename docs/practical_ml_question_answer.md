# One Practical ML Question — Answer

## What is happening?

The training loss came down, which means training itself did not hit an error and went forward
normally. But the validation loss shows that the model's output on data it was not trained on
got noticeably worse, not better. Here I suspect the model did not have enough variety in the
data it saw during this epoch, because its handling of unfamiliar data did not improve — it
actually got worse. The character error rate got worse as well.

Putting those together, the model is overfitting: it is memorising the training data rather
than learning from it.

## What evidence supports my conclusion?

Three numbers, and two of them moved in the wrong direction.

The training loss falling tells me the training process is working, so the problem is not a bug
in the pipeline. The validation loss rising tells me that what the model learned does not carry
over to new data. And the CER rising confirms it, because CER is what actually appears in the
output rather than an internal number.

It is worth noting what a different pattern would have meant. If the validation loss had gone
down in epoch 2 but the CER had gone up, I could suspect either that the dataset contains dirty
data, or that the model can predict the next token correctly when it is given the previous ones
but cannot generate a whole transcript on its own — the loss is measured with teacher forcing
and the CER is not, so they can move independently. But that is not what happened here. Both got
worse while the training loss kept improving, so my reading is that the dataset contains
repeated data and the model has overfitted on some of it.

I would also note that this is only two measurement points, which is early for a firm
conclusion. But both measures moved in the same direction and by a large amount, so the
direction is not in doubt.

## Which change would I try first?

I would stop the training first, so it does not go further in the wrong direction and does not
cost more compute, and so that what has been trained so far is kept.

I would keep the epoch 1 checkpoint, because it is the best model this run produced and because
it gives me a baseline: later, when I apply methods to improve the training, I can compare
against it and see whether the run actually got better.

I would do this first because it costs nothing. The better model already exists on disk, no
retraining is needed, and unlike the other options it cannot make things worse. Changing the
dropout, the learning rate or the capacity all require a full training run before I learn
anything, and each of them can push the model into underfitting if I go too far. Stopping cannot.

Stopping does not fix the cause. It stops the damage while I find the cause.

## What changes would I try next?

After that I would work on the dataset. First I would remove duplicate records and audio files
that are exactly the same, and I would cap transcripts that are identical to each other at a
maximum of three copies. I did this in my own project, so I expect it to address the problem
here as well.

If I change the dataset, I would retrain from scratch rather than continuing from the epoch 1
checkpoint, because that checkpoint was trained on the faulty data and would carry whatever it
learned from it into the new run. Starting from scratch also makes the comparison clean: I can
put the two runs side by side and see which one behaved better.
