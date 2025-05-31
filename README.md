# 12th Level Biology Vector DB and Description Writing

## Customizing the Embedding

The application uses the `intfloat/multilingual-e5-large` embedding model, which requires specific prefixes like `"query"` and `"passage"` for its inputs. To ensure proper functionality, adjustments are needed in the custom embedding class located within the RAPTOR folder.

* **Navigate to the Custom Embedding Class**
  Locate the custom embedding class in the `RAPTOR` directory.
* **Modify the Embedding Function**
  Adjust the embedding function to prefix input texts with `"query: "` or `"passage: "`, as required by the embedding model.
* **Adjust for Different Embeddings (If Needed)**
  If using a different embedding model, update the custom embedding class accordingly. Remove or modify prefixes based on the new model's requirements.
