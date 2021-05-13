import React from "react";
import { GlobalSettings } from "../model/botSettings";

type TextMsgProps = {
  messageText: string;
  isFromBot: boolean;
  showAvatar: boolean;
};

class TextMsg extends React.Component<TextMsgProps> {
  constructor(props: TextMsgProps) {
    super(props);
  }

  render(): React.ReactNode {
    const className = this.props.isFromBot ? "msg left-msg" : "msg right-msg";
    return (
      <div className={className}>
        {this.props.isFromBot && (
          <div
            className="msg-img"
            style={{
              backgroundImage: `url(${GlobalSettings.botImage})`,
              visibility: this.props.showAvatar ? "visible" : "hidden",
            }}
          ></div>
        )}
        <div className="msg-bubble">
          <div className="msg-text">{this.props.messageText}</div>
        </div>
      </div>
    );
  }
}

export default TextMsg;
