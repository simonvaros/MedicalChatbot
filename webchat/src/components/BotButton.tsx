import React from "react";
import { GlobalSettings } from "../model/botSettings";

type BotButtonProps = {
  botButtonClick: () => void;
};

class BotButton extends React.Component<BotButtonProps> {
  constructor(props: BotButtonProps) {
    super(props);
  }

  render(): React.ReactNode {
    return (
      <div id="bot-button" onClick={this.props.botButtonClick}>
        <img src={GlobalSettings.botImage}></img>
      </div>
    );
  }
}

export default BotButton;
