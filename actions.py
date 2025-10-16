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


"""
YES "Hug", "Hit", "Kick", "Push", "Pull", "Slap", "Pat on back", "Step on foot", "Link arms", "Pat on cheek","Touch head"
NO "Wave", "Point finger at","Chase", "Rock-paper-scissors", "Bend", "Chat", "Thumb up","Imitate","Look back","Fly kiss"

G049T000A000R004  Hug
G004T003A002R016  Wave
G044T002A010R004  Slap
G024T004A020R004  Rock-paper-scissors
G026T003A015R000  Step on foot
G041T010A037R013  Look Back
G012T002A011R008  Pat on Back
G028T008A039R018  Fly Kiss
G022T009A032R010  Touch Head
G020T004A024R021  Bend


"""

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