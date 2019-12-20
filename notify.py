"""
Optional support for Slack notifications when jobs are done.
The (secret) webhook URL should be set in the environment variable:
    SLACK_NOTIFY_WEBHOOK=https://hooks.slack.com/services/.../.../...
"""
import os
import json
import socket

class SlackNotifier():
    """
    Since notifications are not crucial in this context, this class is
    designed to fail with a simple print in case anything goes wrong.
    """

    def __init__(self, prefix = None):
        self.enabled = False
        if 'SLACK_NOTIFY_WEBHOOK' in os.environ:
            url = os.environ['SLACK_NOTIFY_WEBHOOK']
            if (url.startswith('https')):
                self.webhook = url
                self.enabled = True
        self.prefix = prefix or '[{}] '.format(socket.gethostname())

    def notify(self, message):
        if not self.enabled:
            print('Slack notification not sent: webhook URL not available. '
                  'Set the `SLACK_NOTIFY_WEBHOOK` environment variable.')
            return False

        try:
            import requests

            r = requests.post(self.webhook, data=json.dumps({
                'text': self.prefix + message
            }), headers={
                'Content-type': 'application/json'
            }, timeout=2)

            if r.status_code == 200:
                return True

            print('Failed to send Slack notification: response code ' + str(r.status_code))
            return True
        except ImportError as e:
            print('Failed to send Slack notification: `requests` '
                  'module not available:\n' + str(e))
        except Exception as e:
            print('Failed to send Slack notification:\n' + str(e))
        return False


if __name__ == '__main__':
    SlackNotifier().notify("Hi there")
