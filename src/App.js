import Home from './components/Home';
import Description from './components/Description'
import About from './components/About';
import { BrowserRouter, Route, Switch, Link } from 'react-router-dom';
import { AmplifySignOut } from '@aws-amplify/ui-react';
import { Toolbar } from '@material-ui/core';
import './App.css';

function App() {
    return (
        <main>
          <BrowserRouter>
            <Toolbar>
                <Link class="App-button" to="/">Home</Link>{' '}
                <Link class="App-button" to='/description'>Family Description</Link>{''}
                <Link class="App-button" to='/about'>ABOUT</Link>{''}
                <AmplifySignOut buttonText="Sign Out" />
            </Toolbar>
            
            <Switch>
                <Route path="/" component={Home} exact />
                <Route path="/description" component={Description} />
                <Route path="/about" component={About} />
            </Switch>
          </BrowserRouter>
        </main>
    );
}

export default App;
