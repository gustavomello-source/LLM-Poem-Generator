import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import datetime

class PoetryModel:
    '''
    Class that represents a poetry model using the Hugging Face transformers library.
    This class is responsible for fine-tuning a pre-trained model on a dataset of poems
    and generating new poems based on a given prompt.

    Attributes:
        model_name (str): The name of the pre-trained model to use.
        device (torch.device): The device to run the model on.
        tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
        model (transformers.AutoModelForCausalLM): The model for generating text.

    Methods:
        __init__(self, model_name): Constructor of the class.
        fine_tune(self, dataloader, num_epochs, learning_rate): Fine-tunes the model on a dataset of poems.
        generate_poem(self, prompt, max_length): Generates a new poem based on a given prompt.
        save_model(self, output_dir): Saves the model and tokenizer to a directory.
    '''

    def __init__(self, model_name="facebook/opt-350m"):
        '''
        Constructor of the PoetryModel class.

        Args:
            model_name (str): The name of the pre-trained model to use.
        
        Raises:
            ValueError: If the model_name is None or an empty string.
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
    def fine_tune(self, dataloader, num_epochs=3, learning_rate=5e-5):
        '''
        Fine-tunes the model on a dataset of poems.

        Args:
            dataloader (torch.utils.data.DataLoader): The DataLoader for the dataset.
            num_epochs (int): The number of epochs to train the model.
            learning_rate (float): The learning rate for the optimizer.
        
        Returns:
            list: A list of dictionaries containing training statistics by epoch.
        '''
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        best_loss = float('inf')
        training_stats = []

        epoch_progress = tqdm(range(num_epochs), desc="Epochs", position=0)

        print('======== Training Started ========')
        print(f'Number of examples: {len(dataloader.dataset)}')
        print(f'Batch size: {dataloader.batch_size}')
        print(f'Total steps per epoch: {len(dataloader)}')
        start_time = time.time()


        for epoch in epoch_progress:
            print(f'\n{"="*20} Epoch {epoch+1}/{num_epochs} {"="*20}')
            
            # Reset metrics for this epoch
            epoch_loss = 0
            batch_times = []
            
            # Create progress bar for batches
            batch_progress = tqdm(dataloader, desc=f"Training", position=1, leave=True)
            batch_start = time.time()
            
            self.model.train()
            
            for batch_idx, batch in enumerate(batch_progress):
                # Process batch
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = input_ids.clone()
                
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                epoch_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update progress bar with current loss
                current_loss = epoch_loss / (batch_idx + 1)
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                avg_batch_time = sum(batch_times) / len(batch_times)
                
                # Estimate time remaining
                remaining_batches = len(dataloader) - (batch_idx + 1)
                remaining_time = remaining_batches * avg_batch_time
                
                # Update progress bar description
                batch_progress.set_description(
                    f"Loss: {current_loss:.4f} | "
                    f"Batch Time: {avg_batch_time:.2f}s | "
                    f"ETA: {str(datetime.timedelta(seconds=int(remaining_time)))}"
                )
                
                batch_start = time.time()
        
            # Calculate average loss for this epoch
            avg_epoch_loss = epoch_loss / len(dataloader)
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                print(f"\n→ Best model so far! Loss: {best_loss:.4f}")
                
            # Calculate epoch statistics
            epoch_time = time.time() - start_time
            training_stats.append({
                'epoch': epoch + 1,
                'avg_loss': avg_epoch_loss,
                'epoch_time': epoch_time,
            })
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'Average Loss: {avg_epoch_loss:.4f}')
            print(f'Time taken: {str(datetime.timedelta(seconds=int(epoch_time)))}')
            
        print("\n======== Training Finished ========")
        print(f"Total training time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
        
        # Print training summary
        print("\nTraining stats by epoch:")
        for stat in training_stats:
            print(f"Epoch {stat['epoch']}: Loss = {stat['avg_loss']:.4f}")

        return training_stats
    
    def generate_poem(self, prompt, max_length=300):
        '''
        Generates a new poem based on a given prompt.

        Args:
            prompt (str): The prompt for the poem.
            max_length (int): The maximum length of the generated poem.

        Returns:
            str: The generated poem.

        Raises:
            None
        '''
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def save_model(self, output_dir):
        '''
        Saves the model and tokenizer to a directory.
        
        Args:
            output_dir (str): The directory to save the model and tokenizer.

        Returns:
            None
        '''
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")