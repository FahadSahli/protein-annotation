import App from "./App";
import config from './aws-exports';
import ReactDOM from "react-dom";
import Amplify from 'aws-amplify';
import "./index.css";


Amplify.configure(config);

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
