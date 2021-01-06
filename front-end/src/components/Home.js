import { SetS3Config } from "./services";
import Predictions from "./Predictions";
import { withAuthenticator } from '@aws-amplify/ui-react';
import Storage from "@aws-amplify/storage";
import { Component } from "react";
import { Auth } from 'aws-amplify';
import './styles.css';

class Home extends Component {
  
  state = {
    fileName: "",
    inputFile: "",
    userID : "",
    response: "",
    showPredictions: false
  };
  
  async componentDidMount() {

    const info = await Auth.currentUserInfo();
    console.log('Returned info: ', info['attributes']['sub']);
    //this.setState({ info });
     this.setState({ userID : info['attributes']['sub']});
  }
  
  uploadFile = () => {
    SetS3Config("", "private", "");
    Storage.put(`${this.state.userID}/${this.upload.files[0].name}`,
                this.upload.files[0],
                { contentType: this.upload.files[0].type })
      .then(result => {
        this.upload = null;
        this.setState({ response: "Successful upload!" });
      })
      .catch(err => {
        this.setState({ response: `Cannot uploading file: ${err}` });
      });
  };
  
  
  render() {
    return (
      <div className="App">
        
        <h1>Protein Annotation</h1>
        
        <h3>Please upload your sequences...</h3>
          <input
            type="file"
            //accept="image/png, image/jpeg"
            style={{ display: "none" }}
            ref={ref => (this.upload = ref)}
            onChange={e =>
              this.setState({
                inputFile: this.upload.files[0],
                fileName: this.upload.files[0].name
              })
            }
          />
        <input value={this.state.fileName} placeholder="Select a CSV file" />
        <button
          onClick={e => {
            this.upload.value = null;
            this.upload.click();
          }}
          loading={this.state.uploading}
          class="button"
        >
          Browse
        </button>
        <button onClick={this.uploadFile} class="button"> Upload File </button>
        {!!this.state.response && <div>{this.state.response}</div>}
        
        <div>
          <button onClick={() => window.location.reload()} class="button">Refresh</button>
          <Predictions userID={this.state.userID} />
        </div>
      </div>
    );
  }
}

export default withAuthenticator(Home);

/*
<button onClick={() => this.setState({ showPredictions: !this.state.showPredictions })} class="button">Show Table</button>
{this.state.showPredictions && <Predictions userID={this.state.userID} />}
*/
