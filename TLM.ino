#include <stdio.h>
#include "esp_heap_caps.h"
#include "Tokenizer.h"
#include "sp_vocab.h"
#include "Tiny.h"
#include <float.h>

String inputString = "";
bool stringComplete = false;

int64_t token_ids[SEQ_LEN];
int num_tokens = 0;

int token(const char* text) {
  tokenize(text, token_ids, &num_tokens);

  // Sliding window protection: keep only last SEQ_LEN tokens to avoid array overflow
  if (num_tokens > SEQ_LEN) {
    int start_idx = num_tokens - SEQ_LEN;
    for (int i = 0; i < SEQ_LEN; i++) {
      token_ids[i] = token_ids[start_idx + i];
    }
    num_tokens = SEQ_LEN;
  }

  // Zero-fill remaining slots
  for (int i = num_tokens; i < SEQ_LEN; i++) {
    token_ids[i] = 0;
  }

  return 0;
}

float (*logits)[SEQ_LEN][VOCAB];
int64_t (*token_ids_2d)[SEQ_LEN] = (int64_t(*)[SEQ_LEN])token_ids;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize fast O(1) hash tokenizer table
  init_tokenizer_hash();

  // Initialize model static memory unions in PSRAM
  init_tiny_model();

  Serial.println("==========================================");
  Serial.println(" ESP32-P4 Accelerated LLM Initialized! ");
  Serial.println("==========================================");
  Serial.printf("Free Internal SRAM: %d bytes\n", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  Serial.printf("Free PSRAM:         %d bytes\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Allocate Logits buffer safely in PSRAM
  logits = (float(*)[SEQ_LEN][VOCAB])heap_caps_malloc(sizeof(float) * SEQ_LEN * VOCAB, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  if (logits == NULL) {
    Serial.println("Error: Could not allocate logits buffer in PSRAM!");
  } else {
    Serial.println("Logits memory allocated successfully in PSRAM.");
  }
}

void print_token_clean(const char* token) {
  if (strlen(token) >= 3 && 
      (unsigned char)token[0] == 0xE2 && 
      (unsigned char)token[1] == 0x96 && 
      (unsigned char)token[2] == 0x81) {
    Serial.print(' ');
    if (strlen(token) > 3) {
      Serial.print(token + 3);
    }
  } else {
    Serial.print(token);
  }
}

void loop() {
  if (stringComplete) {
    inputString.trim();
    if (inputString.length() > 0) {
      Serial.printf("\nPrompt: '%s'\n", inputString.c_str());
      Serial.print("Generating: ");

      String tem = inputString;
      uint32_t start_time = millis();

      for (int t = 0; t < 32; t++) {
        token(tem.c_str());
        
        // Accelerated dual-core forward pass
        forward_pass(token_ids_2d, logits);

        int next_id = sample_next_token(logits, num_tokens, 40, 0.7f);
        if (next_id <= 0) next_id = 0;

        const char* next_str = vocab[next_id];
        print_token_clean(next_str);
        tem += next_str;
      }

      uint32_t elapsed = millis() - start_time;
      float tps = 32.0f / (elapsed / 1000.0f);
      Serial.printf("\n\n--- Stats: 32 tokens in %d ms (%.2f tokens/sec) ---\n", elapsed, tps);
    }
    inputString = "";
    stringComplete = false;
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      if (inputString.length() > 0) {
        stringComplete = true;
      }
    } else {
      inputString += inChar;
    }
  }
}
