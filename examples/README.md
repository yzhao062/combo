### How to run examples?

First combo should be installed or you should download the github repository.
````cmd
pip install combo
pip install --upgrade combo # make sure the latest version is installed!
````

After that, you could simply copy & paste the code or directly run the examples.

---

### Introduction of Examples
Examples are structured as follows:
- Examples are named as XXX_example.py, in which XXX is the model name.
- For all examples, you can find corresponding models at combo/models/

For instance: 
- classifier_comb: classifier_comb_example.py
- cluster_comb: cluster_comb_example.py
- ... other individual algorithms


---

### What if I see "xxx module could be found" or "Unresolved reference"

**First check combo is installed with pip.**

If you have not but simply download the github repository, please make
sure the following codes are presented at the top of the code. The examples 
import the models by relying the code below if combo is not installed:

```python
import sys
sys.path.append("..")
```
This is a **temporary solution** for relative imports in case **combo is not installed**.

If combo is installed using pip, no need to import sys and sys.path.append("..")
Feel free to delete these lines and directly import combo models.

