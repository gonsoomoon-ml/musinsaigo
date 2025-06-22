#!/usr/bin/env python3
"""
Image Description Generator using Amazon Bedrock Claude 3.7
Generates detailed person-focused descriptions for fashion images in metadata.jsonl
"""

import json
import os
import sys
import boto3
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time
import traceback
import textwrap
from botocore.exceptions import ClientError

# Configure timestamp for output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'description_generation_{timestamp}.log'

print(f"Logging initialized. Log file: {log_filename}")
print("=" * 80)
print("Image Description Generation Started")
print("=" * 80)

class BedrockImageDescriptionGenerator:
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize the Bedrock client for Claude 3.7 Sonnet
        
        Args:
            region_name: AWS region where Bedrock is available
        """
        try:
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name
            )
            print(f"Bedrock client initialized for region: {region_name}")
        except Exception as e:
            print(f"Failed to initialize Bedrock client: {e}")
            raise
    
    def generate_description(self, image_path: str, current_text: str = "") -> str:
        """
        Generate detailed image description using Claude 3.7 Sonnet with converse API
        """
        start_time = time.time()
        try:
            print(f"Processing image: {os.path.basename(image_path)}")
            
            # Read image as bytes (like the example)
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Extract image extension and validate format
            image_extension = image_path.split(".")[-1].lower()
            supported_formats = ['jpeg', 'jpg', 'png', 'gif', 'webp']
            if image_extension not in supported_formats:
                raise ValueError(f"Unsupported image format: {image_extension}")
            
            # Map jpg to jpeg for API compatibility
            if image_extension == 'jpg':
                image_extension = 'jpeg'
            
            print(f"Image loaded successfully: {os.path.basename(image_path)} (size: {len(image_bytes)} bytes, format: {image_extension})")
            
            system_prompt = textwrap.dedent("""# Text2Image Model Fine-tuning Caption Generation System Prompt

                ## Role
                You are a professional image caption generator specialized in creating training data for Text2Image model fine-tuning.

                ## Core Guidelines

                ### 1. Person-Centered Description (Top Priority)
                - Prioritize describing **human characteristics** in the image above all else
                - Detail gender, age group, appearance, hairstyle, facial expression, and pose
                - Provide comprehensive descriptions of clothing and accessories worn by the person

                ### 2. Visual Element Priority Order
                1. **Person Description** (gender, age, appearance, expression)
                2. **Clothing & Style** (including colors, materials, design details)
                3. **Pose & Actions**
                4. **Background & Environment** (briefly)
                5. **Overall mood or aesthetic style**

                ### 3. Caption Constraints
                - **Maximum 30 words**
                - Include concise yet essential visual information
                - Exclude unnecessary speculation or interpretation
                - Use objective and descriptive language only

                ### 4. Writing Format
                - Write in English
                - Prefer comma-separated phrase structure
                - Maintain natural sentence flow
                - Include visible brand names or text if present

                ## Example Structure
                "[Gender] [age group] person with [hairstyle], wearing [clothing description], [pose/expression], [background], [overall style/mood]"

                ## Important Notes
                - Do not speculate or identify specific individuals
                - Replace copyrighted characters or brands with generic terms
                - Express inappropriate content in neutral terms
                - Describe only what is clearly visible, avoid assumptions""")

            user_prompt = f"""Analyze this fashion image and create a detailed description following the system guidelines.

Current description: "{current_text}"

Please provide a more detailed and accurate description focusing on the person in the image."""

            # Create message structure with system prompt included in user message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": user_prompt},
                        {
                            "image": {
                                "format": image_extension,
                                "source": {
                                    "bytes": image_bytes
                                }
                            }
                        }
                    ]
                }
            ]

            # System prompts configuration
            system_prompts = [
                {
                    "text": system_prompt
                }
            ]

            # Inference configuration
            inference_config = {
                "maxTokens": 70,
                "temperature": 0.1,
            }

            print(f"Calling Bedrock API for: {os.path.basename(image_path)}")
            
            # Make the API call using converse with system parameter
            response = self.bedrock_runtime.converse(
                modelId="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                messages=messages,
                system=system_prompts,
                inferenceConfig=inference_config
            )

            # Extract the response like the example
            output_message = response['output']['message']
            
            # Get the text content
            response_text = ""
            for content in output_message['content']:
                if 'text' in content:
                    response_text += content['text']
            
            if response_text.strip():
                processing_time = time.time() - start_time
                print(f"\n📝 Caption for {os.path.basename(image_path)} (⏱️ {processing_time:.2f}s):\n   {response_text.strip()}\n")
                
                # Print token usage like the example
                token_usage = response['usage']
                print(f"   Input tokens: {token_usage['inputTokens']}")
                print(f"   Output tokens: {token_usage['outputTokens']}")
                print(f"   Total tokens: {token_usage['totalTokens']}")
                print(f"   Stop reason: {response['stopReason']}\n")
                
                return response_text.strip()
            else:
                processing_time = time.time() - start_time
                print(f"⚠️  No valid response content for {os.path.basename(image_path)} (⏱️ {processing_time:.2f}s)")
                return str(current_text) if current_text else "A person in fashion clothing"

        except ClientError as err:
            processing_time = time.time() - start_time
            
            # Handle ThrottlingException with retry
            if err.response['Error']['Code'] == 'ThrottlingException':
                print(f"\n⚠️  Rate limit exceeded for {os.path.basename(image_path)}. Waiting 60 seconds and retrying...")
                time.sleep(60)
                
                try:
                    print(f"Retrying API call for: {os.path.basename(image_path)}")
                    response = self.bedrock_runtime.converse(
                        modelId="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                        messages=messages,
                        system=system_prompts,
                        inferenceConfig=inference_config
                    )
                    
                    # Extract the response like the example
                    output_message = response['output']['message']
                    
                    # Get the text content
                    response_text = ""
                    for content in output_message['content']:
                        if 'text' in content:
                            response_text += content['text']
                    
                    if response_text.strip():
                        total_processing_time = time.time() - start_time
                        print(f"\n📝 Caption for {os.path.basename(image_path)} (⏱️ {total_processing_time:.2f}s, retried):\n   {response_text.strip()}\n")
                        
                        # Print token usage like the example
                        token_usage = response['usage']
                        print(f"   Input tokens: {token_usage['inputTokens']}")
                        print(f"   Output tokens: {token_usage['outputTokens']}")
                        print(f"   Total tokens: {token_usage['totalTokens']}")
                        print(f"   Stop reason: {response['stopReason']}\n")
                        
                        return response_text.strip()
                    else:
                        total_processing_time = time.time() - start_time
                        print(f"⚠️  No valid response content for {os.path.basename(image_path)} (⏱️ {total_processing_time:.2f}s, retried)")
                        return str(current_text) if current_text else "A person in fashion clothing"
                        
                except Exception as retry_error:
                    total_processing_time = time.time() - start_time
                    error_msg = f"Retry failed for {os.path.basename(image_path)}: {str(retry_error)}"
                    print(f"\n❌ ERROR: {error_msg}")
                    print(f"   File: {image_path}")
                    print(f"   Error Type: {type(retry_error).__name__}")
                    print(f"   Processing Time: {total_processing_time:.2f}s")
                    print(f"   Continuing with original text...\n")
                    return str(current_text) if current_text else "A person in fashion clothing"
            
            # Handle other ClientErrors
            error_msg = f"Client error generating description for {os.path.basename(image_path)}: {err.response['Error']['Message']}"
            print(f"\n❌ ERROR: {error_msg}")
            print(f"   File: {image_path}")
            print(f"   Error Type: {type(err).__name__}")
            print(f"   Processing Time: {processing_time:.2f}s")
            print(f"   Full Traceback:")
            print(f"   {'='*50}")
            traceback.print_exc()
            print(f"   {'='*50}")
            print(f"   Continuing with original text...\n")
            return str(current_text) if current_text else "A person in fashion clothing"
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error generating description for {os.path.basename(image_path)}: {str(e)}"
            print(f"\n❌ ERROR: {error_msg}")
            print(f"   File: {image_path}")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Processing Time: {processing_time:.2f}s")
            print(f"   Full Traceback:")
            print(f"   {'='*50}")
            traceback.print_exc()
            print(f"   {'='*50}")
            print(f"   Continuing with original text...\n")
            return str(current_text) if current_text else "A person in fashion clothing"
    
    def process_metadata_file(self, metadata_path: str, output_path: str = None, num_files: int = None) -> None:
        """
        Process the metadata.jsonl file and generate new descriptions
        
        Args:
            metadata_path: Path to the input metadata.jsonl file
            output_path: Path to the output metadata.jsonl file (optional)
            num_files: Number of files to process (optional, default: process all)
        """
        total_start_time = time.time()
        try:
            # Read the original metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_lines = f.readlines()
            
            # Limit the number of files to process
            if num_files is not None:
                metadata_lines = metadata_lines[:num_files]
                print(f"Processing {len(metadata_lines)} images (limited to {num_files}) from {metadata_path}")
            else:
                print(f"Processing {len(metadata_lines)} images from {metadata_path}")
            
            # Set output path
            if output_path is None:
                output_path = metadata_path.replace('.jsonl', '_updated.jsonl')
            
            # Create or clear the output file
            with open(output_path, 'w', encoding='utf-8') as f:
                pass  # Create empty file
            
            print(f"Output file initialized: {output_path}")
            
            # Process each line and write immediately
            processed_count = 0
            for i, line in enumerate(metadata_lines):
                try:
                    data = json.loads(line.strip())
                    image_path = data['file_name']
                    current_text = data.get('text', '')
                    
                    # Construct full image path
                    full_image_path = os.path.join(
                        os.path.dirname(metadata_path), 
                        image_path
                    )
                    
                    if not os.path.exists(full_image_path):
                        print(f"⚠️  Image not found: {full_image_path}")
                        # Write original data to output file
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            f.flush()  # 즉시 디스크에 쓰기
                            os.fsync(f.fileno())  # 강제 동기화
                        processed_count += 1
                        continue
                    
                    print(f"Processing image {i+1}/{len(metadata_lines)}: {image_path}")
                    
                    # Generate new description
                    new_description = self.generate_description(full_image_path, current_text)
                    
                    # Update the data
                    data['text'] = new_description
                    
                    # Write to output file immediately
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                        f.flush()  # 즉시 디스크에 쓰기
                        os.fsync(f.fileno())  # 강제 동기화
                    
                    processed_count += 1
                    print(f"✅ Saved to output file: {processed_count}/{len(metadata_lines)}")
                    
                    # Add a delay to avoid rate limiting
                    print(f"   Waiting 18 seconds before next API call...")
                    time.sleep(18)
                    
                except Exception as e:
                    error_msg = f"Failed to process line {i+1}: {e}"
                    print(f"\n❌ ERROR: {error_msg}")
                    print(f"   Line: {i+1}")
                    print(f"   Data: {line.strip()[:100]}...")
                    print(f"   Error Type: {type(e).__name__}")
                    print(f"   Full Traceback:")
                    print(f"   {'='*50}")
                    traceback.print_exc()
                    print(f"   {'='*50}")
                    print(f"   Keeping original data...\n")
                    
                    # Write original data to output file
                    try:
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(line.strip() + '\n')
                            f.flush()  # 즉시 디스크에 쓰기
                            os.fsync(f.fileno())  # 강제 동기화
                        processed_count += 1
                        print(f"✅ Saved original data to output file: {processed_count}/{len(metadata_lines)}")
                    except Exception as write_error:
                        print(f"❌ Failed to write to output file: {write_error}")
            
            total_time = time.time() - total_start_time
            print(f"✅ Processing completed!")
            print(f"   Processed: {processed_count}/{len(metadata_lines)} images")
            print(f"   Output saved to: {output_path}")
            print(f"   Total processing time: {total_time:.2f}s")
            
        except Exception as e:
            total_time = time.time() - total_start_time
            print(f"Failed to process metadata file: {e}")
            print(f"Total processing time: {total_time:.2f}s")
            raise

def main():
    """Main function to run the description generator"""
    main_start_time = time.time()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate image descriptions using Amazon Bedrock Claude 3.7',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_descriptions.py
  python generate_descriptions.py -i /path/to/input.jsonl -o /path/to/output.jsonl
  python generate_descriptions.py --input /path/to/input.jsonl --output /path/to/output.jsonl --region us-west-2
  python generate_descriptions.py -n 5
  python generate_descriptions.py -i /path/to/input.jsonl -o /path/to/output.jsonl -n 10
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default="/home/ubuntu/musinsaigo/test_data/metadata.jsonl",
        help='Input metadata.jsonl file path (default: /home/ubuntu/musinsaigo/test_data/metadata.jsonl)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default="/home/ubuntu/musinsaigo/test_data/metadata_updated.jsonl",
        help='Output metadata.jsonl file path (default: /home/ubuntu/musinsaigo/test_data/metadata_updated.jsonl)'
    )
    
    parser.add_argument(
        '-r', '--region',
        type=str,
        default="us-east-1",
        help='AWS region for Bedrock (default: us-east-1)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually running the API calls'
    )
    
    parser.add_argument(
        '-n', '--num-files',
        type=int,
        default=None,
        help='Number of files to process (default: process all files)'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input file
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        if args.dry_run:
            # Read and show what would be processed
            with open(args.input, 'r', encoding='utf-8') as f:
                metadata_lines = f.readlines()
            print(f"DRY RUN: Would process {len(metadata_lines)} images from {args.input}")
            print(f"DRY RUN: Would save results to {args.output}")
            print(f"DRY RUN: Using region {args.region}")
            return
        
        # Initialize the generator
        generator = BedrockImageDescriptionGenerator(region_name=args.region)
        
        # Process the metadata file
        generator.process_metadata_file(args.input, args.output, args.num_files)
        
        total_time = time.time() - main_start_time
        success_msg = "Description generation completed successfully!"
        print(f"\n✅ SUCCESS: {success_msg}")
        print(f"   Input: {args.input}")
        print(f"   Output: {args.output}")
        print(f"   Log file: {log_filename}")
        print(f"   Total execution time: {total_time:.2f}s\n")
        
    except Exception as e:
        total_time = time.time() - main_start_time
        error_msg = f"Script failed: {e}"
        print(f"\n❌ CRITICAL ERROR: {error_msg}")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Total execution time: {total_time:.2f}s")
        print(f"   Full Traceback:")
        print(f"   {'='*50}")
        traceback.print_exc()
        print(f"   {'='*50}")
        print(f"   Please check the log file for more details.")
        print(f"   Log file: {log_filename}\n")
        sys.exit(1)

if __name__ == "__main__":
    main() 