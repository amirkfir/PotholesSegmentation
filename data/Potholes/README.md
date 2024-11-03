
# A Fully Annotated Image Dataset for Pothole Detection

- The directory `annotated-images` contains the images having pothole and their respective annotations (as XML file).
- The file `splits.json` contains the annotation filenames (.xml) of the **training** (80%) and **test** (20%) dataset in following format---

```javascript
{
  "train": ["img-110.xml", "img-578.xml", "img-455.xml", ...],
  "test": ["img-565.xml", "img-498.xml", "img-143.xml", ...]
}
```

### Important Notes

- The `<path>` tag in the xml annotations may not match with your environment, therefore, consider `<filename>` tag.
- The file `splits.json` was generated randomly taking 20% of the total dataset as **test** and the rest as **train**.
