# TestBeam Analysis

Work in progress


## Usage


## LoadBatch

Contains a lot of useful functions that I use to load a `.root` file into Python and to analyze its content

## SensorClasses.py

Basic data structures employed (contained in 'SensorClasses.py'):
### `Batch`
- **Attributes**:
    - `batch_number (int)`: Batch number (example: 403).
    - `angle (float)`: Angle of the DUTs do the beam.
    - `humidity (float)`: Average humidity [%].
    - `temperature (float)`: Average temperature [°C].
    - `S (dict)`: Dictionary containing the two `Oscilloscope`: 

        `{'S1':Oscilloscope1, 'S2':Oscilloscope2}`.

- **Methods**: (for initialization only)
    - `set_fluence_boards()`: manually inserts the fluence values.
    - `set_transimpedance()`: manually sets the transimpedance values [OBSOLETE].

### `Oscilloscope`
- **Attribuses**:
    - `name (str)`: Name of the oscilloscope 'S1' or 'S2'.
    - `channels (dict)`: Dictionary containing the four channels of the `Oscilloscope`:

        `{'Ch1': Sensor1, 'Ch2': Sensor2, 'Ch3': Sensor3, 'Ch4': Sensor4}`.
    - `runs (list)`: List of the runs included in this batch 
    - `tempA (list)`: List of temperatures, from thermometer A
    - `tempB (list)`: List of temperatures, from thermometer B

- **Methods**:
    - `add_sensor(channel, Sensor)`: Sets one of the channels with a `Sensor`.
    - `get_sensor(channel)`: Returns `Sensors` at the called channel.

### `Sensor`
- **Attributes**:
    - `name (str)`: Name assigned to the DUT.
    - `dut_position (int):` Position (order) of the DUT (1-5).
    - `voltage (float)`: Voltage of the DUT [V].
    - `current (float)`: Measured current [A].
    - `board (str)`: Name assigned to the board which the DUT is mounted on.
    - `fluence (str)`: Irradiation of the sensor [#neutron_eq / cm^2].
    - `transimpedance (float)`: Transipedance, depends on the board, it is needed to calculate the charge [mV*ps/fC].


## Single_batch


## Functions description:
Description of the most important functions implemented
- plot()


## ToDo

- [ ] Add information in this README (XD)

<!-- ## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.cern.ch/mpozzess/testbeam-analysis.git
git branch -M master
git push -uf origin master
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.cern.ch/mpozzess/testbeam-analysis/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

***

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers. -->
