import requests
import os
import time
import re

from slackclient import SlackClient

print("BOT_TOKEN : ", os.environ.get("SLACK_BOT_TOKEN"))

class SlackBot:
    def __init__(self, logger):
        self.logger = logger
        self.slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))

        self.HELP_COMMAND = "help"
        self.TRAIN_PLOT = "train"
        self.TRAIN_LOG = "trainlog"
        self.VALID_PLOT = "valid"
        self.VALID_LOG = "validlog"
        self.POSSIBLE_COMMANDS = [self.HELP_COMMAND, self.TRAIN_PLOT, self.TRAIN_LOG, self.VALID_PLOT, self.VALID_LOG]
        
    def run(self):
        READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
        if self.slack_client.rtm_connect(with_team_state=False):
            print("StarterBot connected and running!")
            while True:
                command, channel = self._parse_slack_output(self.slack_client.rtm_read())
                if command and channel:
                    try:
                        self._handle_command(command, channel)
                    except Exception as e:
                        print(e)
                time.sleep(READ_WEBSOCKET_DELAY)
        else:
            print("Connection failed. Invalid Slack token or bot ID?")

    def _parse_slack_output(self, slack_rtm_output):
        """The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
        """
        def parse_direct_mention(message_text):
            MENTION_REGEX = "^<@(|[WU].+?)>(.*)"
            matches = re.search(MENTION_REGEX, message_text)
            # the first group contains the username, the second group contains the remaining message
            return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

        output_list = slack_rtm_output
        if output_list and len(output_list) > 0:
            for output in output_list:
                if output["type"] == "message" and not "subtype" in output:
                    user_id, message = parse_direct_mention(output["text"])
                    return message, output["channel"]
        return None, None



    def _post_img(self, channel, img_path):
        with open(img_path, "rb") as f:
            return requests.post('https://slack.com/api/files.upload',
                                    data={'token': os.environ.get("SLACK_BOT_TOKEN"),
                                            'channels': [channel],
                                            'title': 'Analysis Graph'},
                                    files={'file': f})
            
    def _handle_command(self, command, channel):
        """
            Receives commands directed at the bot and determines if they
            are valid commands. If so, then acts on the commands. If not,
            returns back what it needs for clarification.
        """
        all_commands = "\t".join(self.POSSIBLE_COMMANDS)
        response = "I only support the following commands:[%s]"%(all_commands) 

        if command == self.HELP_COMMAND:
            response = "I support the following commands %s"%(all_commands)

        elif command == self.TRAIN_PLOT:
            img_path = self.logger.log_plot("train", title="Train Log graph", mode="slack")
            self._post_img(channel, img_path)
            return

        elif command == self.VALID_PLOT:
            img_path = self.logger.log_plot("valid", title="Valid Acc graph", mode="slack")
            self._post_img(channel, img_path)
            return

        elif command == self.TRAIN_LOG:
            log_dict = self.logger.log_parse("train")
            response = "Train Log\n"
            for k, v in list(log_dict.items())[-10:]:
                response += "%d, %s\n"%(k, str(v))

        elif command == self.VALID_LOG:
            log_dict = self.logger.log_parse("valid")
            response = "Valid Log\n"
            for k, v in list(log_dict.items())[-10:]:
                response += "%d, %s\n"%(k, str(v))

        self.slack_client.api_call("chat.postMessage", channel=channel,
                            text=response, as_user=True)
