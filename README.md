# MMCompressor

MMcompress is a research project to explore **multi-modal compression** into the **KV cache** of large language models using `<memory>` tokens. It demonstrates how text, images, and eventually video can be compressed into **just one token** — and then **retrieved, reasoned about, or edited** — purely through context.

## 🧠 Core Idea

> Can we simulate "latent memory" with a single token that stands for arbitrary, compressed content — including an image or a set of facts — and recall or manipulate it later? 

MMcompress shows how language models can:
- Compress **images** into one token for later retrieval and description.
- Store and **overwrite** arbitrary **text facts** while retaining others.
- Extend this idea to **video** (ongoing work).

---

## 📦 Setup

```bash
bash env.sh
```

---

## 📁 Project Structure

```bash
MMcompress/
├── text_compressor/     # Compresses and edits structured text facts
├── image_compressor/    # Stores ~500-token images in a single memory token
├── video_compressor/    # WIP: Extending ideas to temporal content
├── env.sh               # Environment setup script
```
Also includes a setup file for GH200 GPU ARM-based.
---

## 📝 Text Compression Example

Store and overwrite **arbitrary facts** in latent memory — even compose across them:

<img src="images/text_compress.png" alt="Text compression example" width="700"/>

---

## 🌲 Image Compression Example

Compress and recall **image tokens (~500)** using **one memory token**:

<img src="images/image_compress.png" alt="Image compression example" width="700"/>

---

## 🧪 Research Directions

- Memory token chaining (`<memory1>`, `<memory2>`) for sequential reasoning
- Dynamic memory replacement, retention, and summarization
- Multimodal context management (text-image-video)
- Applications to retrieval-augmented generation, agent memory, and beyond

---

## 📍 Status

- ✅ **Text** working with small dataset of 10K examples!
- ✅ **Image** working with small dataset of 20K images!
- 🔄 **Video** memory is under development, but has promising results when chunking a video into 10 seconds increments. 

---

## 📫 

Feel free to reach out if you're interested in collaborating or experimenting further!
