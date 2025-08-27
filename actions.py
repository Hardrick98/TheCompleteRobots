import joblib
import argparse
import random

actions_dict = {
    0: "Hug",
    1: "Handshake",
    2: "Wave",
    3: "Grab",
    4: "Hit",
    5: "Kick",
    6: "Posing",
    7: "Push",
    8: "Pull",
    9: "Sit on leg",
    10: "Slap",
    11: "Pat on back",
    12: "Point finger at",
    13: "Walk towards",
    14: "Knock over",
    15: "Step on foot",
    16: "High-five",
    17: "Chase",
    18: "Whisper in ear",
    19: "Support with hand",
    20: "Rock-paper-scissors",
    21: "Dance",
    22: "Link arms",
    23: "Shoulder to shoulder",
    24: "Bend",
    25: "Carry on back",
    26: "Massaging shoulder",
    27: "Massaging leg",
    28: "Hand wrestling",
    29: "Chat",
    30: "Pat on cheek",
    31: "Thumb up",
    32: "Touch head",
    33: "Imitate",
    34: "Kiss on cheek",
    35: "Help up",
    36: "Cover mouth",
    37: "Look back",
    38: "Block",
    39: "Fly kiss"
}



def pick_random_action(action):

  actions = joblib.load("actions.pkl")

  n = random.randint(0, len(actions[action]))

  a = actions[action][n]

  return a

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-a","--action")
  args = parser.parse_args()


  a = pick_random_action(args.action)
  print(a)