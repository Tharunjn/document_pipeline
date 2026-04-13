import sys
import os
import re
import base64
import json
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc.base import ImageRefMode
import requests

# Setup logging
log_file = Path("./extraction_debug.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

def is_significant_image(image_path, min_width=10, min_height=10, min_size_kb=25):
    """
    Check if image is significant (not just text or small graphics).

    Args:
        image_path: Path to image file
        min_width: Minimum width in pixels
        min_height: Minimum height in pixels
        min_size_kb: Minimum file size in KB

    Returns:
        True if image should be kept, False to exclude
    """
    try:
        # Check file size
        file_size_kb = image_path.stat().st_size / 1024
        if file_size_kb < min_size_kb:
            logger.info(f"Excluding {image_path.name}: Too small ({file_size_kb:.1f}KB)")
            return False

        # Check dimensions
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        width, height = img.size

        if width < min_width or height < min_height:
            logger.info(f"Excluding {image_path.name}: Small dimensions ({width}x{height})")
            return False

        # Check if image is mostly white/empty (simple graphics/text boxes)
        import numpy as np
        img_array = np.array(img.convert('RGB'))

        # Calculate white/light pixel percentage
        white_pixels = np.sum((img_array[:, :, 0] > 240) &
                             (img_array[:, :, 1] > 240) &
                             (img_array[:, :, 2] > 240))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        white_percentage = (white_pixels / total_pixels) * 100

        if white_percentage > 80:  # More than 80% white = likely just text/graphics
            logger.info(f"Excluding {image_path.name}: Mostly white ({white_percentage:.1f}% empty)")
            return False

        logger.info(f"✓ Keeping {image_path.name}: {width}x{height}, {file_size_kb:.1f}KB")
        return True

    except Exception as e:
        logger.error(f"Error checking {image_path.name}: {e}")
        return True  # Keep if we can't determine


def get_vlm_summary(image_bytes, context_text="", vlm_url="http://10.117.100.61:11434/api/chat", model="qwen3-vl:8b"):
    """
    Send image to locally hosted VLM and get summary with context.
    Assumes Ollama with vision model like llava.

    Args:
        image_bytes: Image data as bytes
        context_text: Surrounding text from document for context
        vlm_url: VLM API endpoint
        model: Model name
    """
    img_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # Build prompt with context
    if context_text:
        prompt = f"""Please analyze this image. Here is the context from the document where this image appears:

<context>
{context_text.strip()}
</context>

Based on the context, provide a concise summary or description of what this image shows and how it relates to the surrounding text. Focus on the key information and insights the image provides."""
    else:
        prompt = "Describe this image in detail."

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64]
            }
        ],
        "stream": False
    }
    try:
        logger.info(f"Getting VLM summary with context ({len(context_text)} chars)...")
        logger.debug(f"CONTEXT TEXT: {context_text}")
        response = requests.post(vlm_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        summary = result['message']['content']
        context_text.strip()
        
        logger.debug(f"VLM response: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error getting VLM summary: {e}")
        return "Image summary not available."

def extract_and_summarize_images(doc, output_folder):
    """
    Extract images from docling document using document model.
    Capture surrounding text context for better VLM summaries.
    Filter out small/insignificant images, then summarize.
    Returns list of (image_ref, summary) for replacement.
    """
    images_info = []
    extracted_dir = output_folder / 'extracted_images'
    filtered_dir = output_folder / 'filtered_images'

    extracted_dir.mkdir(exist_ok=True)
    filtered_dir.mkdir(exist_ok=True)

    global_img_counter = [0]  # Use list to pass by reference

    try:
        # Try to access pages and their content
        if hasattr(doc, 'pages'):
            for page_num, page in enumerate(doc.pages, 1):
                logger.info(f"Processing page {page_num}...")

                # Get all items/blocks from page
                if hasattr(page, 'children'):
                    page_items = list(page.children)

                    for item_idx, item in enumerate(page_items):
                        item_type = item.__class__.__name__ if hasattr(item, '__class__') else 'Unknown'

                        # Check if item is an image
                        if 'Picture' in item_type:
                            try:
                                # Try to get image data
                                if hasattr(item, 'image') and item.image:
                                    img_data = None

                                    # Try different ways to access image
                                    if hasattr(item.image, 'data'):
                                        img_data = item.image.data
                                    elif hasattr(item.image, 'uri') and item.image.uri.startswith('data:'):
                                        # Extract from data URI
                                        uri = item.image.uri
                                        if ',' in uri:
                                            img_data = base64.b64decode(uri.split(',')[1])
                                    elif hasattr(item.image, '__bytes__'):
                                        img_data = bytes(item.image)

                                    if img_data:
                                        # Use unique incrementing counter
                                        img_filename = f"image_{global_img_counter[0]}.png"
                                        global_img_counter[0] += 1

                                        # Save to extracted folder first
                                        extracted_path = extracted_dir / img_filename
                                        with open(extracted_path, 'wb') as f:
                                            f.write(img_data if isinstance(img_data, bytes) else img_data.encode())
                                        logger.info(f"Extracted: {img_filename}")

                                        # Extract nearby context (items before and after this image)
                                        context_texts = []

                                        # Look backwards for text (up to 5 items)
                                        for prev_idx in range(max(0, item_idx - 5), item_idx):
                                            prev_item = page_items[prev_idx]
                                            prev_type = prev_item.__class__.__name__ if hasattr(prev_item, '__class__') else 'Unknown'

                                            # Debug: print item type
                                            logger.debug(f"[Context] Looking back at idx {prev_idx}, type: {prev_type}")

                                            # Get text from various item types
                                            text_found = None

                                            # Try to extract text
                                            if hasattr(prev_item, 'text') and prev_item.text:
                                                text_found = prev_item.text
                                            elif hasattr(prev_item, 'get_text'):
                                                try:
                                                    text_found = prev_item.get_text()
                                                except:
                                                    pass

                                            if text_found:
                                                if 'Heading' in prev_type:
                                                    context_texts.append(f"[Section: {text_found}]")
                                                    logger.debug(f"Found heading: {text_found}")
                                                elif 'Table' in prev_type:
                                                    context_texts.append(f"[Table: {text_found[:200]}...]")
                                                    logger.debug(f"Found table")
                                                else:
                                                    context_texts.append(text_found)
                                                    logger.debug(f"Found text: {text_found}")

                                        # Look forward for caption/text (up to 2 items)
                                        for next_idx in range(item_idx + 1, min(len(page_items), item_idx + 3)):
                                            next_item = page_items[next_idx]
                                            next_type = next_item.__class__.__name__ if hasattr(next_item, '__class__') else 'Unknown'

                                            logger.debug(f"[Caption] Looking forward at idx {next_idx}, type: {next_type}")

                                            # Try to extract text
                                            text_found = None
                                            if hasattr(next_item, 'text') and next_item.text:
                                                text_found = next_item.text
                                            elif hasattr(next_item, 'get_text'):
                                                try:
                                                    text_found = next_item.get_text()
                                                except:
                                                    pass

                                            if text_found:
                                                if len(text_found) < 200:
                                                    context_texts.append(f"[Caption: {text_found}]")
                                                    logger.debug(f"Found caption: {text_found}")
                                                    break
                                                else:
                                                    context_texts.append(text_found)
                                                    logger.debug(f"Found text: {text_found}")
                                                    break

                                        # Build context string
                                        context = '\n'.join(context_texts)
                                        # if len(context) > 800:
                                        #     context = context[-800:]

                                        logger.info(f"Context length: {len(context)} chars")
                                        if context:
                                            logger.info(f"Context preview: {context}")
                                        else:
                                            logger.warning("No context found")

                                        # Store with context for later
                                        images_info.append({
                                            'filename': img_filename,
                                            'extracted_path': extracted_path,
                                            'context': context,
                                            'page': page_num
                                        })
                            except Exception as e:
                                logger.error(f"Error processing image: {e}")

        # If no images found via pages, try export_to_dict
        if not images_info:
            logger.info("No images found in pages, trying JSON export...")
            doc_json = doc.export_to_dict()

            # Process JSON while maintaining context order
            # We'll track the "current context" as we traverse
            context_buffer = []  # Stack of text segments we've seen
            MAX_CONTEXT_CHARS = 2000

            def find_images_with_context(obj, depth=0):
                """Traverse JSON, maintain context, extract images"""
                if depth > 20:
                    return []

                found = []

                if isinstance(obj, dict):
                    # Check for text content first - add to context
                    if 'text' in obj and isinstance(obj['text'], str) and obj['text'].strip():
                        text_content = obj['text'].strip()
                        context_buffer.append(text_content)

                        # Print debug info about what section we're in
                        logger.debug(f"[Section] {text_content}")

                    # Now check for image in this dict
                    for key, value in obj.items():
                        if isinstance(value, str) and value.startswith('data:image/'):
                            try:
                                if ',' in value:
                                    img_data = base64.b64decode(value.split(',')[1])
                                    img_filename = f"image_{global_img_counter[0]}.png"
                                    global_img_counter[0] += 1

                                    extracted_path = extracted_dir / img_filename
                                    with open(extracted_path, 'wb') as f:
                                        f.write(img_data)
                                    logger.info(f"Extracted: {img_filename}")

                                    # Build context from context buffer
                                    context_str = '\n'.join(context_buffer)

                                    # Limit to MAX_CONTEXT_CHARS, keeping most recent content
                                    # if len(context_str) > MAX_CONTEXT_CHARS:
                                    #     context_str = context_str[-MAX_CONTEXT_CHARS:]

                                    if context_str:
                                        logger.info(f"✓ Context ({len(context_str)} chars): Full context captured")
                                        logger.debug(f"Context content: {context_str}")
                                    else:
                                        logger.warning("Context: (empty)")

                                    found.append({
                                        'filename': img_filename,
                                        'extracted_path': extracted_path,
                                        'context': context_str,
                                        'page': 0
                                    })
                            except Exception as e:
                                logger.error(f"Error extracting image: {e}")

                    # Recursively process nested dicts
                    for key, value in obj.items():
                        if key != 'text':  # Don't reprocess text keys
                            found.extend(find_images_with_context(value, depth + 1))

                elif isinstance(obj, list):
                    # Process list items in order
                    for item in obj:
                        found.extend(find_images_with_context(item, depth + 1))

                return found

            images_info.extend(find_images_with_context(doc_json))

        # Filter and summarize extracted images with context
        logger.info("\nFiltering and summarizing images...")
        final_images = []
        rejected_images = []

        for img_info in images_info:
            extracted_path = img_info['extracted_path']

            # Check if file still exists
            if not extracted_path.exists():
                logger.warning(f"Skipping {img_info['filename']}: File not found")
                rejected_images.append((img_info['filename'], "Image file not found"))
                continue

            # Filter out insignificant images
            if not is_significant_image(extracted_path, min_width=80, min_height=80, min_size_kb=25):
                logger.info(f"Rejected {img_info['filename']}")
                rejected_images.append((img_info['filename'], "Image filtered out - too small or insignificant"))
                continue

            # Copy to filtered folder
            filtered_path = filtered_dir / img_info['filename']
            import shutil
            shutil.copy2(extracted_path, filtered_path)
            logger.info(f"Accepted and copied: {img_info['filename']}")

            # Get summary for significant images with context
            try:
                with open(filtered_path, 'rb') as f:
                    image_data = f.read()
                logger.info(f"Getting summary for {img_info['filename']}...")
                if img_info['context']:
                    logger.debug(f"Context: {img_info['context']}")
                else:
                    logger.warning(f"Context: (none found)")

                summary = get_vlm_summary(image_data, context_text=img_info['context'])
                final_images.append((img_info['filename'], summary))
            except Exception as e:
                logger.error(f"Error summarizing {img_info['filename']}: {e}")
                final_images.append((img_info['filename'], "Image summary not available"))

    except Exception as e:
        logger.error(f"Error extracting images: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info(f"Total significant images: {len(final_images)}")
    logger.info(f"Total rejected images: {len(rejected_images)}")

    # Combine for markdown replacement: accepted images with summaries, rejected with placeholders
    all_images_for_replacement = final_images + rejected_images
    return all_images_for_replacement

def replace_images_in_md(md_content, images_info, output_folder):
    """
    Replace image references in Markdown with:
    1. Links to filtered images + summaries (for accepted images)
    2. Placeholder text (for rejected images)
    """
    # Separate accepted vs rejected images
    accepted_images = []
    rejected_images = []

    for filename, summary in images_info:
        # Check if this image is in filtered_images folder
        filtered_path = output_folder / 'filtered_images' / filename
        if filtered_path.exists():
            accepted_images.append((filename, summary))
        else:
            rejected_images.append((filename, summary))

    logger.info(f"Replacement prep: {len(accepted_images)} accepted, {len(rejected_images)} rejected")
    logger.debug(f"Accepted images: {[f for f, _ in accepted_images]}")
    logger.debug(f"Rejected images: {[f for f, _ in rejected_images]}")

    # Create counter to track replacement order
    img_index = [0]  # Use list to pass by reference

    # Replace all HTML image comments with appropriate content
    def replace_image_comment(match):
        if img_index[0] < len(images_info):
            filename, summary = images_info[img_index[0]]
            current_idx = img_index[0]
            img_index[0] += 1

            # Check if this is an accepted or rejected image
            filtered_path = output_folder / 'filtered_images' / filename
            if filtered_path.exists():
                # Accepted image - include file reference and summary
                logger.info(f"Replacing image {current_idx} ({filename}) with filtered reference + summary")
                return f"![Image](./filtered_images/{filename})\n\n**Summary:** {summary}"
            else:
                # Rejected image - just placeholder
                logger.info(f"Replacing image {current_idx} ({filename}) with rejection reason")
                return f"<!-- Image filtered out: {summary} -->"

        logger.warning(f"Image index {img_index[0]} exceeds images_info length {len(images_info)}")
        return match.group(0)

    # Replace HTML image comments (format: <!-- image -->)
    original_count = len(re.findall(r'<!-- image -->', md_content))
    logger.info(f"Found {original_count} image placeholder comments in markdown")

    md_content = re.sub(
        r'<!-- image -->',
        replace_image_comment,
        md_content
    )

    logger.info(f"Replaced {len(accepted_images)} accepted images with summaries")
    logger.info(f"Replaced {len(rejected_images)} rejected images with placeholders")

    return md_content

def smart_chunk_markdown(md_content, output_folder):
    """
    Perform smart chunking: split by headers, and handle tables separately.
    """
    chunks_dir = output_folder / 'chunks'
    chunks_dir.mkdir(exist_ok=True)
    
    lines = md_content.split('\n')
    chunks = []
    current_chunk = []
    
    for line in lines:
        if re.match(r'^#{1,4}\s', line):  # Header line
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        current_chunk.append(line)
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Now, for each chunk, check for tables and split further
    final_chunks = []
    for chunk in chunks:
        if '|' in chunk and '\n|' in chunk:  # Likely has table
            # Split by table
            parts = re.split(r'(\n\|.*\|(?:\n\|.*\|)*\n)', chunk)
            for part in parts:
                if part.strip():
                    final_chunks.append(part.strip())
        else:
            final_chunks.append(chunk)
    
    # Save chunks
    for i, chunk in enumerate(final_chunks):
        with open(chunks_dir / f'chunk_{i}.txt', 'w', encoding='utf-8') as f:
            f.write(chunk)

def main(input_file, output_folder):
    """
    Main function to process DOCX or PDF.
    """
    input_path = Path(input_file)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file {input_file} does not exist.")
        return

    # Determine format
    if input_path.suffix.lower() == '.pdf':
        format_options = PdfPipelineOptions()
        format_options.do_ocr = True  # Enable OCR for PDFs
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: format_options
            }
        )
    else:
        converter = DocumentConverter()

    # Convert document
    logger.info(f"Converting {input_path.name}...")
    result = converter.convert(str(input_path))
    doc = result.document

    # Extract images and get summaries
    logger.info("Extracting images...")
    images_info = extract_and_summarize_images(doc, output_path)
    logger.info(f"Extracted {len(images_info)} images")

    # Export to Markdown
    logger.info("Generating Markdown...")
    md_content = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)

    # Debug: Check for image references in markdown
    import re
    image_refs = re.findall(r'!\[.*?\]\([^)]+\)', md_content)
    logger.info(f"Found {len(image_refs)} image references in markdown:")
    for ref in image_refs[:5]:  # Show first 5
        logger.debug(f"  {ref[:100]}")

    # Replace images with file references and summaries
    if images_info:
        md_content = replace_images_in_md(md_content, images_info, output_path)
        logger.info(f"Replaced images in markdown")
    else:
        logger.warning("Warning: No images extracted!")

    # Save the full Markdown
    with open(output_path / 'document.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    logger.info(f"Saved Markdown to {output_path / 'document.md'}")

    # Perform smart chunking
    logger.info("Chunking document...")
    smart_chunk_markdown(md_content, output_path)

    logger.info(f"✓ Processing complete. Output in {output_folder}")
    logger.info(f"Log file: {log_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Usage: python process_doc.py <input_file> <output_folder>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    main(input_file, output_folder)
