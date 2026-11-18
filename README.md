# Does Enzyme Function Transfer Across the Tree of Life?

This is your task document. Read it before you start, and come back to it whenever you are not sure what to do next. It explains the science, the schedule, how to work in Google Colab, the rules for using AI, and what to hand in.

## Contents

- [What you will do and why](#what-you-will-do-and-why)
- [Background](#background)
- [The question](#the-question)
- [Before day 1](#before-day-1)
- [Schedule](#schedule)
- [How to open a notebook in Colab](#how-to-open-a-notebook-in-colab)
- [How the notebook cells work](#how-the-notebook-cells-work)
- [AI use](#ai-use)
- [Deliverables](#deliverables)
- [Stretch goals (optional)](#stretch-goals-optional)
- [How to open your pull request](#how-to-open-your-pull-request)
- [Troubleshooting](#troubleshooting)
- [How to ask for help](#how-to-ask-for-help)

## What you will do and why
 You will run a complete experiment yourself, from raw data to a final answer.

Here is the plan:

1. **Explore the data.** You will look at thousands of enzymes from five species: human, mouse, fruit fly, baker's yeast, and the bacterium *E. coli*.
2. **Build a baseline.** A baseline is a simple model that you compare other models against. Yours uses ESM-2, a transformer trained on proteins, without changing it.
3. **Fine-tune ESM-2.** You will train ESM-2 further on human enzymes, so it learns to predict what kind of chemical reaction each enzyme speeds up.
4. **Evaluate.** You will test both models on all five species and check whether fine-tuning made a real difference.
5. **Interpret.** You will explain the pattern you find using biology, including looking up specific proteins.

Scientists use these same steps when they want to know whether a model learned something real.

The task takes about 10 hours, spread over several sessions on different days. You will mostly work on your own, in Google Colab in your web browser. 

You do not need to know Python yet. Notebook 00 teaches the Python you need. It compares each idea to Java, and it uses the enzyme data for its examples, so you also get to know the data.

## Background

Read this section once before you start. You do not need to memorize it. Each notebook reminds you of what you need when you need it.

### Proteins and enzymes

A **protein** is a molecule that does a job in a living cell. It is a long chain of small building blocks called **amino acids**. There are 20 kinds of amino acids, and each one has a one-letter code. So a protein can be written as a string of letters, much like a Java `String`. For example, human salivary amylase starts like this:

```
MKLFWLLFTIGFCWAQYSSNTQQGRTSIVH...
```

Each letter in the chain is called a **residue**. The order of the residues decides how the chain folds up, and the folded shape decides what the protein does. The instructions for making a protein are stored in a **gene**, a stretch of DNA (the molecule that stores a cell's instructions).

An **enzyme** is a protein that speeds up a chemical reaction. Without enzymes, most reactions in your body would be far too slow to keep you alive. Amylase, the enzyme above, is in your saliva. It breaks starch into sugars.

### EC numbers and the seven enzyme classes

Scientists label enzymes by the reaction they speed up, using an **EC number** (EC stands for Enzyme Commission). An EC number has four parts, such as `3.2.1.1` for amylase. You read it from left to right, from general to specific. The first number is the enzyme's **class**. Each later number narrows it down, until the last one names the exact reaction.

There are seven classes:

| Class | Name | What enzymes in this class do |
|---|---|---|
| 1 | Oxidoreductases | Move electrons from one molecule to another. Chemists call this oxidation and reduction. |
| 2 | Transferases | Move a small chemical group, such as a phosphate group, from one molecule to another. |
| 3 | Hydrolases | Break a bond by adding a water molecule. Amylase is one. |
| 4 | Lyases | Break a bond without adding water and without moving electrons. They often leave behind a double bond or a ring. |
| 5 | Isomerases | Rearrange the atoms inside one molecule, turning it into a different form of itself. |
| 6 | Ligases | Join two molecules together, using energy from ATP, the molecule cells use to carry energy. |
| 7 | Translocases | Move ions (charged atoms or molecules) or other molecules across a membrane, the thin layer that wraps a cell and some of the compartments inside it. |

Your models will predict only the class, the first number of the EC number. That class is the protein's **label**: the right answer the model is trying to predict. Two details:

- Some EC numbers are incomplete, like `3.4.-.-`. The dashes mean the later parts are not known yet. These proteins keep their class (here, class 3).
- A few proteins speed up reactions from more than one class. Each protein needs exactly one label, so those proteins are left out.

The classes are not equally common. You will count how many of each there are in Notebook 01.

### Where the data comes from

All proteins come from **UniProt** (uniprot.org), a free public database of proteins. You will use only **Swiss-Prot**, the part of UniProt where experts have checked each entry by hand, so the EC numbers can be trusted. Each protein has an **accession**, a short ID such as `P0DUB6` (that one is salivary amylase). You can type an accession into the search box at uniprot.org to read about that protein.

The data was downloaded on 2026-11-18 (UniProt release 2026_04) and is saved in this repository, so you never have to download it yourself.

### Species, orthologs, and evolutionary distance

A **species** is one kind of living thing, such as humans or house mice. You will work with five species. The models learn only from human proteins. The other four species are used only for testing.

| Species | What it is | Role in this task | Shared an ancestor with humans about |
|---|---|---|---|
| Human (*Homo sapiens*) | Us | Training, validation, and testing | Not applicable |
| Mouse (*Mus musculus*) | A mammal, like us | Testing only | 87 million years ago |
| Fruit fly (*Drosophila melanogaster*) | An insect | Testing only | 694 million years ago |
| Baker's yeast (*Saccharomyces cerevisiae*) | A single-celled fungus, the kind used to make bread rise | Testing only | 1,275 million years ago |
| *E. coli* (*Escherichia coli*, lab strain K-12) | A bacterium: a single cell with no nucleus (the compartment that holds the DNA in human, mouse, fly, and yeast cells) | Testing only | 4,290 million years ago |

The last column shows roughly how long ago each species and humans shared a **common ancestor**: a species that lived long ago, which both later species descend from. This is called the **divergence time**. It is one way to measure **evolutionary distance**: the longer ago two species split, the more distant they are. The four test species form a ladder that gets more distant at each step: a mammal, an insect, a fungus, and a bacterium. The times are estimates from TimeTree (timetree.org), a public collection of published studies, and they are saved in `data/divergence_times.tsv`.

When one species splits into two, both new species inherit the same genes. Over millions of years, each copy slowly changes through **mutations**, small random changes in DNA. Two genes in different species that came from the same gene in their common ancestor are called **orthologs**. The word is also used for the proteins those genes make. Orthologs often keep the same job, even after their sequences have changed.

The longer ago two species split, the more time their orthologs have had to change. But a protein can change many of its letters and still do the same reaction. That is the puzzle at the heart of this task.

### ESM-2, a protein language model

A **language model** is a model that learns the patterns of a language by reading huge amounts of text. **ESM-2** does the same thing with proteins. It is a transformer that read millions of protein sequences. During its training, some amino acids in each sequence were hidden, and the model had to guess them from the rest. To get good at that, it had to learn patterns in how proteins are built.

ESM-2 comes in several sizes. You will use the smallest one, `facebook/esm2_t6_8M_UR50D`. It has 6 transformer layers and about 8 million **parameters**, the numbers inside a model that change when it is trained. You will download it from **Hugging Face**, a website where researchers share trained models.

For each residue in a protein, ESM-2 produces a list of 320 numbers. To get one list for the whole protein, you take the average of those lists over all of its residues. That list of 320 numbers is the protein's **embedding**: a summary of the protein that a computer can compare and learn from.

ESM-2 can read at most 1,024 **tokens** (pieces of input) at a time. For ESM-2, each residue is one token, and two extra tokens mark the start and the end. So any protein longer than 1,022 residues is cut short, which is called **truncation**. Notebook 01 shows how many proteins this affects.

### The baseline and fine-tuning

You will compare two ways of using ESM-2.

**Baseline (Notebook 02).** ESM-2 stays **frozen**: its parameters never change. You compute an embedding for every protein. Then you train a **logistic regression** on the human training proteins. Logistic regression is a simple **classifier**, a model that sorts things into groups. It learns how to turn the 320 numbers of an embedding into a guess of the enzyme class.

**Fine-tuning (Notebook 03).** You start from the same ESM-2, add a small classification layer on top, and keep training the whole model on the same human training proteins and their labels. This time, all 8 million parameters can change. That is what **fine-tuning** means: taking a model that was trained for one job (guessing hidden amino acids) and training it more for a new job (predicting the enzyme class).

So here is the difference. The baseline learns only a small classifier on top of embeddings that never change. Fine-tuning also changes ESM-2 itself, so the embeddings change too.

To keep the comparison fair, both models start from the same ESM-2, learn from the same training proteins, and are tested on the same proteins. That way, any difference between them comes from fine-tuning, not from different data.

To compare models, you will give each one a **score** on each species: a number that says how well its predictions match the true labels. Notebook 02 explains which score you will use and why.

You will also see a third, very simple "model" called the **floor**: always guess the class that is most common in the training proteins. It shows what you get with no learning at all.

### Training, validation, and test sets

The human proteins are split into three groups, so you can check the models on proteins they have never seen:

- **Training set** (about 70% of human proteins): the models learn from these.
- **Validation set** (about 15%): used during fine-tuning to choose settings and to pick the best saved version of the model. A saved version is called a **checkpoint**.
- **Test set** (about 15%): kept aside until the end. Its score is the human result you report.

You use the validation set to make choices, so its score can look better than it really is. The test set gives a fair score. All mouse, fly, yeast, and *E. coli* proteins are test proteins too. The models never train on them.

## The question

The title of this task mentions the **tree of life**: the branching family tree that connects all living things. Here is the question you will answer:

> A protein language model is fine-tuned to recognize enzyme function from human protein sequences. Does it recognize the same function in other organisms, and how does that ability change with evolutionary distance from human?

(An **organism** is a living thing. In this task, it means one of the five species.)

Here are two ways to think about it. Will a model trained on human enzymes work best on close relatives, whose orthologs have had less time to change? Or will it learn something about each kind of reaction that is general enough to work across the whole tree of life? The answer could be one of these, a mix of both, or something else. Your experiment will show you, and the notebooks will help you work out where and why.

By the end, you should be able to say, with evidence from your own results:

1. how much fine-tuning changed the predictions compared with the baseline, if at all,
2. for which species and which enzyme classes it made a difference, and for which it did not, and
3. a plausible biological reason for that pattern.

Nobody expects a particular answer. A clear, honest result that you can explain is a success, whatever it shows.

## Before day 1

Do these steps before your first session, so that session can start right away.

- [ ] **1. A personal Google account that can open Colab.** Go to colab.research.google.com, sign in, and make a new notebook. In the first cell, type `print("hello")` and press Shift+Enter. If you see `hello` appear, it works. Use a personal account, not a school account: school accounts often block Colab. Your Google account also comes with Google Drive, where the notebooks will save your work.
- [ ] **2. A GitHub account.** GitHub is a website that stores code projects and their history. If you do not have an account, sign up for free at github.com.
- [ ] **3. Your private copy.** A **repository** is a project folder on GitHub that keeps every version of its files. The original repository for this task is public, which means anyone can see it. You will work in your own private copy of it, so only you and your mentor can see your work and your answers. To make it:
  1. Open the original repository's page on GitHub. Your mentor will send you the link.
  2. Click the green **Use this template** button, then **Create a new repository**.
  3. Leave **Include all branches** unchecked.
  4. Under **Owner**, choose your own account. For **Repository name**, type a name such as `intern-training`.
  5. Choose **Private**.
  6. Click **Create repository**.

  Do not click **Fork** instead. A fork is a different kind of copy, and a fork of a public repository is always public, so your answers would be visible to everyone, including future interns.
- [ ] **4. Your mentor added as a collaborator.** A **collaborator** is someone you give access to your private repository, so your mentor can see and review your work. In your private copy:
  1. Click **Settings** (at the top of the page).
  2. In the menu on the left, click **Collaborators**. GitHub may ask for your password again.
  3. Click **Add people**, type your mentor's GitHub username (ask your mentor for it), and send the invitation.

  Your mentor will get an email and accept it.
- [ ] **5. A working branch.** A **branch** is a separate line of work inside a repository. The starting branch is called `main`. You will save all of your work on your own branch, never on `main`. At the end, a pull request asks to bring your branch's changes into `main`. To make your branch:
  1. On your private copy's main page, click the button that says `main`, just above the list of files.
  2. Type a name for your branch, such as `my-work`.
  3. Click the option to create the branch. It says something like **Create branch my-work from main**.

  Remember this name. You will use it every session.

## Schedule

The task is about 10 hours. Do the sessions in order. Each of the first six sessions covers one notebook. The notebooks are in `notebooks/`. Each one opens by telling you what the step is for and how long it should take.

The times are guides, not limits. If a session runs long, that is normal. Mention it at your next check-in.

| Session | Notebook | What you do | Runtime | Time |
|---|---|---|---|---|
| 1 | Setup and `00_python_primer.ipynb` | Open Colab, connect your Google Drive, and learn the Python you need, using the enzyme data. | CPU | 1 hour 15 minutes |
| 2 | `01_explore_data.ipynb` | Count how many enzymes of each class each species has, make a histogram of protein lengths, and check the step that removes proteins with more than one class. | CPU | 1 hour |
| 3 | `02_baseline.ipynb` | Compute embeddings with frozen ESM-2, train a logistic regression on human proteins, make predictions, and score them. | T4 GPU | 1 hour 15 minutes |
| | **Check-in 1** with your mentor | Talk through the data and the baseline (see below). | | 15 minutes |
| 4 | `03_fine_tune.ipynb` | Choose the number of epochs and the learning rate, run the training, plot the loss curve, and save your model. This includes waiting while it trains. | T4 GPU | 1 hour 15 minutes |
| 5 | `04_evaluate.ipynb` | Score every model on every species, make confusion matrices, and make the headline figure. | CPU | 1 hour 15 minutes |
| | **Check-in 2** with your mentor | Talk through your evaluation (see below). | | 15 minutes |
| 6 | `05_interpret.ipynb` | Make UMAP plots, write answers to the interpretation questions, and look up proteins on UniProt. | T4 GPU | 1 hour 30 minutes |
| 7 | Presentation and pull request | Make and give a presentation of about 10 minutes. **Check-in 3** is the question-and-answer time after it. Then open your pull request. | | 2 hours |
| | **Total** | | | **10 hours** |

New words in this table:

- **Runtime**: the computer that Google lends you to run a notebook. "CPU" means a computer with only a CPU (central processing unit), the normal kind of processor. "T4 GPU" means a computer that also has a GPU (graphics processing unit), a chip that does many math operations at once, which makes neural networks much faster. T4 is the name of the GPU model that free Colab offers. The next section shows how to switch.
- **Histogram**: a bar chart that shows how many values fall into each range.
- **Epoch**: one full pass through all of the training proteins.
- **Learning rate**: how big a step the model takes each time it updates its parameters.
- **Loss curve**: a plot of the **loss**, a number that measures how wrong the model's guesses are, as training goes on.
- **Confusion matrix**: a table that shows, for each true class, how often the model predicted each class.
- **Headline figure**: the main plot of the project. It shows each model's score on each species, plotted against that species' divergence time from humans.
- **UMAP plot**: UMAP is the dimension reduction method from your presentations. It squashes each 320-number embedding down to two numbers, so every protein can be drawn as a dot on a flat plot. You will make UMAP plots before and after fine-tuning.

### Check-ins

Three times during the task. Can happen in-person or over email. Your mentor will ask you follow up questions to check for understanding. Each check-in also has one "change one thing" question: you predict what a small change will do, then run it to see.

1. **Check-in 1, after Notebook 02:** the data, the label, how common each class is, embeddings, and what logistic regression does with them.
2. **Check-in 2, after Notebook 04:** how the fine-tuned model compares with the baseline, how you measured that, and how the results differ from species to species.
3. **Check-in 3, the questions after your presentation:** your interpretation, including evolutionary distance, the result for each species, which classes transferred, and your UMAP plots.

You answer check-in questions without AI. If something is shaky at check-in 1 or 2, that is not a grade. It tells you and your mentor what to review before the next session.

## How to open a notebook in Colab

**Google Colab** is a free website that runs notebooks on Google's computers. A **notebook** is a document made of **cells**. A text cell holds writing. A code cell holds Python code that you can run, one cell at a time. When you run a code cell, its output (printed text or a plot) appears right below it.

### Open a notebook

1. Go to colab.research.google.com and sign in with your personal Google account.
2. Click **File > Open notebook**.
3. Click the **GitHub** tab.
4. Check the box **Include private repos**. The first time, a pop-up window asks you to let Colab use your GitHub account. Sign in to GitHub and approve it.
5. In the search box, type your GitHub username and press Enter.
6. Under **Repository**, choose your private copy. Under **Branch**, choose your working branch (for example `my-work`), not `main`.
7. Click the notebook you want, for example `notebooks/00_python_primer.ipynb`.
8. If Colab warns that the notebook was not written by Google, click **Run anyway**. The notebook came from your own copy.
9. For Notebooks 02, 03, and 05, switch to a GPU before you run anything: **Runtime > Change runtime type**, choose **T4 GPU**, and click **Save**.
10. Run the first cell, the setup cell: click on it and press Shift+Enter (or click the play button on its left). Then run the Configuration cell just below it. The Configuration cell asks to connect Google Drive: click **Connect to Google Drive**, choose your personal account, and allow access.

### What the setup and Configuration cells do

Every notebook starts with a setup cell. It downloads the data and helper code for the task and installs the Python packages it needs. The Configuration cell just below it connects your Google Drive. Things that take a long time to make, such as embeddings and your trained model, are saved in your Drive in `MyDrive/esm2-enzyme-transfer/`, in a folder for your run. Later notebooks load them from there.

So if Colab resets, or you come back on a new day, you only need to rerun the first cells of the notebook you are working on. You do not have to redo earlier notebooks.

### Save your work back to GitHub

Colab does not save your notebook to GitHub by itself. To save it:

1. Click **File > Save a copy in GitHub**.
2. Under **Repository**, choose your private copy. Under **Branch**, choose your working branch.
3. Check that the file path matches where the notebook lives, for example `notebooks/01_explore_data.ipynb`. That way your version replaces the starting version on your branch.
4. In **Commit message**, write a short note about what you did, such as `Finish class counts in Notebook 01`. A **commit** is a saved snapshot of your files, with a message that says what changed.
5. Click **OK**.

Save at the end of every session, and whenever you finish a big step. Always use **Save a copy in GitHub**. Other save options put the notebook somewhere your mentor cannot see it.

### Every session, in short

1. Open the notebook from your working branch.
2. For Notebooks 02, 03, and 05, switch the runtime to **T4 GPU**.
3. Run the setup cell, then the Configuration cell below it, and connect Google Drive when it asks.
4. Work through the cells in order, from top to bottom.
5. Save with **File > Save a copy in GitHub**, to your working branch.
6. When you are done, click **Runtime > Disconnect and delete runtime**, so you do not use up free GPU time while you are away. Your notebook stays saved in GitHub, and your saved files stay in Drive.

## How the notebook cells work

### Cells where you write code

Most of the code is written for you, with comments that explain it. Where you write code, the cell is small, usually 1 to 5 lines, and marked `TODO`. It tells you in plain words what to do and gives a hint that names the exact function to use. Replace the placeholder (such as `None` or `# YOUR CODE HERE`) with your code.

### Check cells

Right after each `TODO` cell is a check cell. Run it. It prints one of two messages:

- `Check passed: ...` means you are on track. Move on.
- `Not yet: ...` means something needs fixing. The message tells you what it expected. Fix your code, rerun your `TODO` cell, and then rerun the check.

"Not yet" is normal. It is how you find out quickly, instead of an hour later. Do not move on past a "Not yet", because later cells use your answer.

Check cells look at the shape, type, or range of your answer, or recompute it from your own variables. They do not contain the expected results.

### Prediction cells

Before each big result, a text cell asks you to write down what you expect and why. Write it before you run the next cell. Later, another cell asks you to compare your prediction with what actually happened.

- Write your prediction yourself, in your own words, without AI.
- A wrong prediction is fine. Figuring out why it was wrong is some of the best learning in this task.
- Do not go back and change a prediction after you see the result.

### Written answers

Some text cells say **Your answer:** (when you edit the cell, it looks like `**Your answer:**`). Double-click the cell to edit it, type your answer after that line, then press Shift+Enter to finish. Write in full sentences, in your own words.

### Messages about missing files

Some notebooks need a file that an earlier notebook makes. If that file is missing, the notebook stops and tells you which notebook makes it. See [Troubleshooting](#troubleshooting).

## AI use

AI coding help is allowed in this task. That includes Colab's built-in AI assistant and chat tools. Using AI well is a real skill, and you are welcome to practice it here. The rules:

1. **Log every use in [`AI_LOG.md`](AI_LOG.md).** Add one row each time you use AI help: the date, the notebook, what you asked, what you kept, and what you changed or rejected. "Nothing" is a fine answer for the last column if it is true. An honest log matters more than a short one.
2. **Check-ins are answered without AI.** AI can write code and even explanations, so the notebooks are where you learn, and the check-ins are where you show what you understand. If AI writes a line you cannot explain, ask it (or your mentor) to explain it until you can.
3. **Write your predictions before you see results,** and write them yourself.
4. **Write your answers in your own words.** You may ask AI to explain an idea, but the answer cells should sound like you.

Your mentor may ask about your predictions and your AI log at check-ins.

To add a row to `AI_LOG.md`:

1. On GitHub, go to your private copy and switch to your working branch with the button that says `main`.
2. Open `AI_LOG.md`.
3. Click the pencil icon (**Edit this file**).
4. Add a new line at the bottom of the table, following the example row.
5. Click **Commit changes**, make sure it will commit to your working branch, and confirm.

You can also keep notes during a session and add all the rows at the end of it.

## Deliverables

When you finish, you hand in these five things:

- [ ] **Completed notebooks 00 to 05** that run from top to bottom without errors, saved to your working branch. A good final test is **Runtime > Restart session and run all**, which reruns every cell in order. It can take a while for the notebooks that use a GPU.
- [ ] **The headline figure** (from Notebook 04) **and the UMAP figures** (from Notebook 05). They are saved inside your notebooks. To use a figure in your slides, right-click it in Colab and save or copy the image.
- [ ] **A presentation of about 10 minutes**, in the same format as your previous three. One possible outline: the question, the data, the two models, the headline figure, what you think explains the pattern, and what you would try next.
- [ ] **Your AI-use log**, `AI_LOG.md`, filled in and saved on your working branch.
- [ ] **A pull request** in your private copy that contains all of the above, for your mentor to review.

## Stretch goals (optional)

Only try these if you finish everything else early. They are at the end of Notebook 05, in this order:

1. **Ortholog check.** For human enzymes that have an ortholog among the yeast enzymes, find the yeast protein whose embedding is most similar. Is it in the same **ortholog group**, a set of proteins that the eggNOG database marks as orthologs of each other? Compare before and after fine-tuning.
2. **Bigger model.** Repeat the main comparison with a larger ESM-2 model, `facebook/esm2_t12_35M_UR50D`, which has about 35 million parameters instead of 8 million.
3. **Sequence identity.** **Sequence identity** is the percent of residues that match when two sequences are lined up. Group the test proteins by their identity to the most similar human training protein, and ask whether the difference between the baseline and the fine-tuned model depends on it.

## How to open your pull request

A **pull request** asks to merge (combine) the changes on your branch into `main`. It shows every change on one page, and your mentor can comment on any line. You open it inside your private copy, so nothing becomes public.

1. Make sure everything is saved on your working branch: all six notebooks (with **File > Save a copy in GitHub**) and `AI_LOG.md`.
2. Go to your private copy on github.com.
3. Click the **Pull requests** tab, then **New pull request**.
4. Set **base** to `main` and **compare** to your working branch. GitHub shows your changes below.
5. Click **Create pull request**.
6. Give it a title, such as `ESM-2 enzyme transfer: notebooks, figures, and AI log`. In the description, list what is included and tell your mentor where to find your slides (for example, a share link).
7. Click **Create pull request** again.
8. If you see **Reviewers** on the right side, choose your mentor. Either way, send your mentor the link to the pull request.

If your mentor asks for changes, make them on the same branch and save them the same way. The pull request updates by itself. Do not merge the pull request yourself. Your mentor reviews and approves it first.

## Troubleshooting

### "Runtime disconnected" or Colab reset

Colab disconnects if you leave it idle for a while, and free sessions have a time limit. Anything that was only in memory is lost, but your saved files in Drive are not.

1. Click **Reconnect** (or **Connect**, at the top right).
2. If the notebook uses a GPU, check that the runtime is still **T4 GPU**.
3. Rerun the setup cell.
4. Rerun the cells between the setup cell and the place where you stopped, in order. A quick way: click the cell where you stopped and choose **Runtime > Run before**.

### No GPU available

Free Colab GPUs are not always available. You may see a message such as "Cannot connect to GPU backend".

- You can keep going on a CPU runtime (choose it under **Runtime > Change runtime type**, or click **Connect without GPU** if Colab offers it). Every notebook still runs on a CPU, just more slowly. The embedding and training steps take the longest.
- Or try again later. GPU access often comes back, sometimes not until the next day.
- Tell your mentor. Your mentor has catch-up files (saved embeddings and a trained model) so you can keep going without waiting.

### Out of memory

You may see "CUDA out of memory" (the GPU ran out of memory) or "Your session crashed after using all available RAM" (the normal memory ran out).

1. Click **Runtime > Restart session**.
2. Rerun the setup cell, then the cells you need, in order.
3. Close notebooks you are not using. Click **Runtime > Manage sessions** and end any sessions you do not need, because each open session uses memory and GPU time.
4. If it happens again at the same cell, tell your mentor which cell and copy the message.

### "Missing ... It is made by Notebook NN"

Each notebook uses files that earlier notebooks saved in your Drive. If one is missing, you will see a message like this:

```
Missing predictions_baseline.tsv. It is made by Notebook 02 (baseline). Run that notebook first, or ask your mentor for the catch-up files.
```

To fix it, open the notebook the message names, run it to the end so it saves the file, then come back. Or ask your mentor for the catch-up files. Also check that you are signed in to the same Google account as before, because each account has its own Drive.

### A check keeps saying "Not yet"

1. Read the whole message. It says what it expected.
2. Look at your variable. Put its name on the last line of a code cell, or use `print(...)`, and run the cell.
3. Reread the hint in the `TODO` cell. It names the exact function to use.
4. Ask AI for help, and log it in `AI_LOG.md`.
5. Still stuck? Ask your mentor (see below).

### I cannot find my private copy in Colab

- Make sure **Include private repos** is checked in the GitHub tab.
- If no pop-up appeared, your browser may have blocked it. Allow pop-ups for colab.research.google.com and check the box again.
- Search for your GitHub username, not the name of the original repository.

### My saved work is gone

- You may have opened the notebook from `main` instead of your working branch. Open it again and choose your branch under **Branch**.
- You may have saved it to Drive by mistake. **Save a copy in Drive** puts it in a folder called `Colab Notebooks` in your Drive. Open it from there, then use **File > Save a copy in GitHub**.

## How to ask for help

Asking for help is a normal part of science. Scientists ask each other for help all the time.

**When:** if you have tried for about 20 minutes and you are still stuck, ask. Do not lose a whole session to one problem.

**How:** message your mentor the way you usually do. Include:

- which notebook and which cell (a screenshot helps),
- what you were trying to do,
- what you already tried,
- the full error message, copied as text (the last few lines of the red text are the most useful), and
- a link to your notebook on your working branch. Your mentor can open it because you added them as a collaborator.

If you find a problem with the starting materials themselves, such as a typo or a cell that breaks for everyone, tell your mentor too. Fixes go back into the original repository so future interns benefit.

You can do this. Take it one cell at a time.
