/*
  Audio Classification Wristband Controller
  
  This Arduino code controls a wristband with 2 vibration motors based on
  audio classification results from Raspberry Pi.
  
  Motor Control Mapping:
  - "soo" sound (class 1) → Top motor vibrates
  - "hum" sound (class 2) → Bottom motor vibrates  
  - "hmm" sound (class 3) → Both motors vibrate simultaneously
  - "disturbance" (class 0) → No vibration (ignored)
  
  Communication: Serial and WiFi
  Hardware: ESP32/Arduino with 2 vibration motors
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>

// Pin definitions
const int TOP_MOTOR_PIN = 5;      // PWM pin for top motor
const int BOTTOM_MOTOR_PIN = 6;   // PWM pin for bottom motor
const int LED_PIN = 13;           // Status LED
const int BATTERY_PIN = A0;       // Battery voltage monitoring

// WiFi configuration
const char* WIFI_SSID = "YourWiFiNetwork";
const char* WIFI_PASSWORD = "YourWiFiPassword";
const int UDP_PORT = 8080;

// Motor control parameters
const int MAX_INTENSITY = 255;    // Maximum PWM value
const int MIN_INTENSITY = 50;     // Minimum PWM value for noticeable vibration
const int DEFAULT_DURATION = 500; // Default vibration duration (ms)
const int MAX_DURATION = 2000;    // Maximum vibration duration (ms)

// Communication objects
WiFiUDP udp;
char packetBuffer[512];

// System state
struct MotorState {
  bool topMotorActive;
  bool bottomMotorActive;
  int topMotorIntensity;
  int bottomMotorIntensity;
  unsigned long topMotorEndTime;
  unsigned long bottomMotorEndTime;
};

MotorState motorState = {false, false, 0, 0, 0, 0};

// Statistics
unsigned long totalCommands = 0;
unsigned long lastCommandTime = 0;
float batteryVoltage = 0.0;

void setup() {
  Serial.begin(9600);
  
  // Initialize pins
  pinMode(TOP_MOTOR_PIN, OUTPUT);
  pinMode(BOTTOM_MOTOR_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BATTERY_PIN, INPUT);
  
  // Turn off motors initially
  analogWrite(TOP_MOTOR_PIN, 0);
  analogWrite(BOTTOM_MOTOR_PIN, 0);
  
  // Initialize WiFi
  setupWiFi();
  
  // Start UDP
  udp.begin(UDP_PORT);
  
  // Startup indication
  startupSequence();
  
  Serial.println("=== Audio Classification Wristband Ready ===");
  Serial.println("Listening for commands from Raspberry Pi...");
}

void loop() {
  // Check for serial commands
  checkSerialCommands();
  
  // Check for WiFi commands
  checkWiFiCommands();
  
  // Update motor states
  updateMotors();
  
  // Update status LED
  updateStatusLED();
  
  // Monitor battery
  monitorBattery();
  
  // Small delay for stability
  delay(10);
}

void setupWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("WiFi connection failed. Using serial only.");
  }
}

void checkSerialCommands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.length() > 0) {
      processSerialCommand(command);
    }
  }
}

void processSerialCommand(String command) {
  // Expected format: "class,confidence,intensity,duration"
  // Example: "1,0.85,200,800"
  
  int commaIndex1 = command.indexOf(',');
  int commaIndex2 = command.indexOf(',', commaIndex1 + 1);
  int commaIndex3 = command.indexOf(',', commaIndex2 + 1);
  
  if (commaIndex1 > 0 && commaIndex2 > 0 && commaIndex3 > 0) {
    int classId = command.substring(0, commaIndex1).toInt();
    float confidence = command.substring(commaIndex1 + 1, commaIndex2).toFloat();
    int intensity = command.substring(commaIndex2 + 1, commaIndex3).toInt();
    int duration = command.substring(commaIndex3 + 1).toInt();
    
    executeVibrationCommand(classId, confidence, intensity, duration);
  } else {
    Serial.println("Invalid command format");
  }
}

void checkWiFiCommands() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
    if (len > 0) {
      packetBuffer[len] = 0;
      processWiFiCommand(String(packetBuffer));
    }
  }
}

void processWiFiCommand(String jsonCommand) {
  // Parse JSON command
  DynamicJsonDocument doc(512);
  DeserializationError error = deserializeJson(doc, jsonCommand);
  
  if (error) {
    Serial.print("JSON parse error: ");
    Serial.println(error.c_str());
    return;
  }
  
  int classId = doc["class"];
  float confidence = doc["confidence"];
  int intensity = doc.containsKey("intensity") ? doc["intensity"] : 128;
  int duration = doc.containsKey("duration") ? doc["duration"] : DEFAULT_DURATION;
  
  executeVibrationCommand(classId, confidence, intensity, duration);
}

void executeVibrationCommand(int classId, float confidence, int intensity, int duration) {
  // Update statistics
  totalCommands++;
  lastCommandTime = millis();
  
  // Validate parameters
  intensity = constrain(intensity, MIN_INTENSITY, MAX_INTENSITY);
  duration = constrain(duration, 100, MAX_DURATION);
  
  // Adjust intensity based on confidence
  int adjustedIntensity = (int)(intensity * confidence);
  adjustedIntensity = constrain(adjustedIntensity, MIN_INTENSITY, MAX_INTENSITY);
  
  Serial.print("Command: Class=");
  Serial.print(classId);
  Serial.print(", Confidence=");
  Serial.print(confidence, 3);
  Serial.print(", Intensity=");
  Serial.print(adjustedIntensity);
  Serial.print(", Duration=");
  Serial.println(duration);
  
  // Execute vibration based on class
  switch (classId) {
    case 0: // Disturbance - ignore
      Serial.println("Disturbance detected - ignoring");
      break;
      
    case 1: // "soo" sound - top motor
      activateTopMotor(adjustedIntensity, duration);
      Serial.println("Activating TOP motor for 'soo' sound");
      break;
      
    case 2: // "hum" sound - bottom motor
      activateBottomMotor(adjustedIntensity, duration);
      Serial.println("Activating BOTTOM motor for 'hum' sound");
      break;
      
    case 3: // "hmm" sound - both motors
      activateBothMotors(adjustedIntensity, duration);
      Serial.println("Activating BOTH motors for 'hmm' sound");
      break;
      
    default:
      Serial.print("Unknown class: ");
      Serial.println(classId);
      break;
  }
}

void activateTopMotor(int intensity, int duration) {
  motorState.topMotorActive = true;
  motorState.topMotorIntensity = intensity;
  motorState.topMotorEndTime = millis() + duration;
  
  analogWrite(TOP_MOTOR_PIN, intensity);
}

void activateBottomMotor(int intensity, int duration) {
  motorState.bottomMotorActive = true;
  motorState.bottomMotorIntensity = intensity;
  motorState.bottomMotorEndTime = millis() + duration;
  
  analogWrite(BOTTOM_MOTOR_PIN, intensity);
}

void activateBothMotors(int intensity, int duration) {
  // Activate both motors simultaneously
  activateTopMotor(intensity, duration);
  activateBottomMotor(intensity, duration);
}

void updateMotors() {
  unsigned long currentTime = millis();
  
  // Check top motor
  if (motorState.topMotorActive && currentTime >= motorState.topMotorEndTime) {
    analogWrite(TOP_MOTOR_PIN, 0);
    motorState.topMotorActive = false;
    motorState.topMotorIntensity = 0;
  }
  
  // Check bottom motor
  if (motorState.bottomMotorActive && currentTime >= motorState.bottomMotorEndTime) {
    analogWrite(BOTTOM_MOTOR_PIN, 0);
    motorState.bottomMotorActive = false;
    motorState.bottomMotorIntensity = 0;
  }
}

void updateStatusLED() {
  // Blink LED to show system is alive
  static unsigned long lastBlink = 0;
  static bool ledState = false;
  
  unsigned long currentTime = millis();
  
  if (motorState.topMotorActive || motorState.bottomMotorActive) {
    // Solid LED when motors are active
    digitalWrite(LED_PIN, HIGH);
  } else if (currentTime - lastBlink > 1000) {
    // Slow blink when idle
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
    lastBlink = currentTime;
  }
}

void monitorBattery() {
  static unsigned long lastBatteryCheck = 0;
  unsigned long currentTime = millis();
  
  if (currentTime - lastBatteryCheck > 10000) { // Check every 10 seconds
    int batteryReading = analogRead(BATTERY_PIN);
    batteryVoltage = (batteryReading * 5.0) / 1024.0; // Convert to voltage
    
    if (batteryVoltage < 3.3) { // Low battery warning
      Serial.print("WARNING: Low battery voltage: ");
      Serial.println(batteryVoltage, 2);
      
      // Flash LED rapidly for low battery
      for (int i = 0; i < 6; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(100);
        digitalWrite(LED_PIN, LOW);
        delay(100);
      }
    }
    
    lastBatteryCheck = currentTime;
  }
}

void startupSequence() {
  Serial.println("Starting wristband initialization...");
  
  // Test both motors briefly
  Serial.println("Testing top motor...");
  analogWrite(TOP_MOTOR_PIN, 150);
  delay(300);
  analogWrite(TOP_MOTOR_PIN, 0);
  delay(200);
  
  Serial.println("Testing bottom motor...");
  analogWrite(BOTTOM_MOTOR_PIN, 150);
  delay(300);
  analogWrite(BOTTOM_MOTOR_PIN, 0);
  delay(200);
  
  Serial.println("Testing both motors...");
  analogWrite(TOP_MOTOR_PIN, 150);
  analogWrite(BOTTOM_MOTOR_PIN, 150);
  delay(300);
  analogWrite(TOP_MOTOR_PIN, 0);
  analogWrite(BOTTOM_MOTOR_PIN, 0);
  
  // LED startup sequence
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
  
  Serial.println("Wristband initialization complete!");
}

void printStatus() {
  Serial.println("=== Wristband Status ===");
  Serial.print("Total commands received: ");
  Serial.println(totalCommands);
  Serial.print("Last command: ");
  Serial.print((millis() - lastCommandTime) / 1000);
  Serial.println(" seconds ago");
  Serial.print("Battery voltage: ");
  Serial.println(batteryVoltage, 2);
  Serial.print("WiFi status: ");
  Serial.println(WiFi.status() == WL_CONNECTED ? "Connected" : "Disconnected");
  Serial.print("Top motor: ");
  Serial.println(motorState.topMotorActive ? "ACTIVE" : "OFF");
  Serial.print("Bottom motor: ");
  Serial.println(motorState.bottomMotorActive ? "ACTIVE" : "OFF");
  Serial.println("========================");
}

// Handle serial commands for debugging
void serialEvent() {
  while (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input == "status") {
      printStatus();
    } else if (input == "test") {
      startupSequence();
    } else if (input.startsWith("test_motor")) {
      // Test specific motor: "test_motor 1 200 1000" (motor, intensity, duration)
      int space1 = input.indexOf(' ', 11);
      int space2 = input.indexOf(' ', space1 + 1);
      
      if (space1 > 0 && space2 > 0) {
        int motor = input.substring(11, space1).toInt();
        int intensity = input.substring(space1 + 1, space2).toInt();
        int duration = input.substring(space2 + 1).toInt();
        
        if (motor == 1) {
          activateTopMotor(intensity, duration);
        } else if (motor == 2) {
          activateBottomMotor(intensity, duration);
        } else if (motor == 3) {
          activateBothMotors(intensity, duration);
        }
      }
    }
  }
}
