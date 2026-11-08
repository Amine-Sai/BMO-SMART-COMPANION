#include <Arduino.h>
#include <ESP32Servo.h>
#include "limits.h"

struct ServoChannel {
  const char* id;
  int pin;
  float min_deg;
  float max_deg;
  float current_deg;
  float target_deg;
  Servo servo;
  bool attached;
};

static ServoChannel servos[4] = {
  {"pan", 18, SERVO_PAN_MIN_DEG, SERVO_PAN_MAX_DEG, SERVO_PAN_BOOT_DEG, SERVO_PAN_BOOT_DEG, Servo(), false},
  {"tilt", 19, SERVO_TILT_MIN_DEG, SERVO_TILT_MAX_DEG, SERVO_TILT_BOOT_DEG, SERVO_TILT_BOOT_DEG, Servo(), false},
  {"left_arm", 21, SERVO_LEFT_ARM_MIN_DEG, SERVO_LEFT_ARM_MAX_DEG, SERVO_LEFT_ARM_BOOT_DEG, SERVO_LEFT_ARM_BOOT_DEG, Servo(), false},
  {"right_arm", 22, SERVO_RIGHT_ARM_MIN_DEG, SERVO_RIGHT_ARM_MAX_DEG, SERVO_RIGHT_ARM_BOOT_DEG, SERVO_RIGHT_ARM_BOOT_DEG, Servo(), false}
};

static unsigned long last_cmd_ms = 0;
static unsigned long last_update_ms = 0;
static String input_buffer = "";

void attach_servo(ServoChannel& s) {
  if (!s.attached) {
    s.servo.attach(s.pin);
    s.attached = true;
  }
}

void detach_all() {
  for (int i = 0; i < 4; i++) {
    if (servos[i].attached) {
      servos[i].servo.detach();
      servos[i].attached = false;
    }
  }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) {
    attach_servo(servos[i]);
    servos[i].servo.write((int)servos[i].current_deg);
  }
  last_cmd_ms = millis();
  last_update_ms = millis();
}

void process_command(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd == "PING") {
    last_cmd_ms = millis();
    for (int i = 0; i < 4; i++) {
      attach_servo(servos[i]);
    }
    Serial.println("PONG");
    return;
  }

  if (cmd == "STOP") {
    detach_all();
    Serial.println("OK STOP");
    return;
  }

  if (cmd.startsWith("MOVE ")) {
    int first_space = cmd.indexOf(' ');
    int second_space = cmd.indexOf(' ', first_space + 1);
    if (second_space == -1) {
      Serial.println("ERR invalid_args");
      return;
    }
    String id = cmd.substring(first_space + 1, second_space);
    String angle_str = cmd.substring(second_space + 1);
    float angle = angle_str.toFloat();

    ServoChannel* target = nullptr;
    for (int i = 0; i < 4; i++) {
      if (id == servos[i].id) {
        target = &servos[i];
        break;
      }
    }

    if (!target) {
      Serial.println("ERR unknown_servo");
      return;
    }

    if (angle < target->min_deg) angle = target->min_deg;
    if (angle > target->max_deg) angle = target->max_deg;

    target->target_deg = angle;
    attach_servo(*target);
    last_cmd_ms = millis();
    Serial.print("OK ");
    Serial.print(id);
    Serial.print(" ");
    Serial.println(angle, 1);
    return;
  }

  Serial.println("ERR unknown_command");
}

void loop() {
  unsigned long now = millis();

  if (now - last_cmd_ms > SERVO_WATCHDOG_TIMEOUT_MS) {
    detach_all();
  }

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      process_command(input_buffer);
      input_buffer = "";
    } else if (c != '\r') {
      input_buffer += c;
    }
  }

  float dt_s = (now - last_update_ms) / 1000.0f;
  if (dt_s >= 0.02f) {
    last_update_ms = now;
    float max_step = SERVO_MAX_DEG_PER_SEC * dt_s;
    for (int i = 0; i < 4; i++) {
      if (servos[i].attached) {
        float diff = servos[i].target_deg - servos[i].current_deg;
        if (abs(diff) <= max_step) {
          servos[i].current_deg = servos[i].target_deg;
        } else if (diff > 0) {
          servos[i].current_deg += max_step;
        } else {
          servos[i].current_deg -= max_step;
        }
        servos[i].servo.write((int)servos[i].current_deg);
      }
    }
  }
}
