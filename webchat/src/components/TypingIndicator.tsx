import React from "react";
import { GlobalSettings } from "../model/botSettings";

class TypingIndicator extends React.Component {
  render(): React.ReactNode {
    return (
      <div className="msg left-msg" id="typing">
        <div
          className="msg-img"
          style={{
            backgroundImage: `url(${GlobalSettings.botImage})`,
          }}
        ></div>
        <div className="msg-bubble">
          <div className="ticontainer">
            <div className="tiblock">
              <div className="tidot"></div>
              <div className="tidot"></div>
              <div className="tidot"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default TypingIndicator;
