## Module 6.1: Sourcing Data Online

### Overview

In this lesson, we will delve into the fundamental aspects of sourcing data from online platforms. This includes understanding the terms and conditions of datasets, extracting data from HTML tables using Pandas, and working with APIs. These skills will form the foundation for the more advanced data sourcing techniques we will explore in the following sessions.

### Class Objectives

By the end of today's class, students will be able to:

* Read and understand data source terms and conditions and licensing agreements.

* Extract data from HTML tables using the Pandas method `from_html()`.

* Make `GET` requests with the Request library.

* Convert JSON into a Python dictionary.

* Convert JSON into a Pandas DataFrame.

---

### Instructor Notes

In this lesson students will build the foundational skills needed to source data for training their AI models. The first section of the lesson focuses on making students conscious of how and where they source their data in two ways. Firstly, students will be introduced to various licensing agreements and urged to be conscious of terms of use for any dataset they intend to use in their own projects. Secondly, students will be encouraged to be discerning about the sources they get their data from (e.g. government organizations, research institutes) to ensure that the data they use is of good quality. Although these two competencies are covered explicitly in the first part of the lesson, the instructor should routinely remind students to be cognizant of licensing terms and reliability of data sources throughout the remainder of the lesson. 

The second part of the lesson focuses on introducing students to online data sources. The students will learn to retrieve data from HTML tables on a Wikipedia page and they will be introduced to APIs. The bulk of today’s lesson will focus on introducing students to APIs and allowing them to practice coding applications that rely on data from API requests. The ability to retrieve data from APIs will be the key skill underpinning the remaining lessons in this module, so it is crucial that students master the `GET` requests and understand how to work with JSON objects to retrieve specific data.

---

### Class Slides

The slides for this lesson can be viewed on Google Drive here: [Module 6.1 Slides](https://docs.google.com/presentation/d/1aDWR9GLDu9F8HpXU40MrUXWpgb2qfoyfsQKfUovPq2I/edit?usp=sharing).

To add the slides to the student-facing repository, download the slides as a PDF by navigating to File, selecting "Download as," and then choosing "PDF document." Then, add the PDF file to your class repository along with other necessary files. You can view instructions for this [here](https://docs.google.com/document/d/1XM90c4s9XjwZHjdUlwEMcv2iXcO_yRGx5p2iLZ3BGNI/edit).

**Note:** Editing access is not available for this document. If you wish to modify the slides, create a copy by navigating to File and selecting "Make a copy...".

---

### Time Tracker

| Start Time | Number | Activity                                           | Duration |
| ---------- | ------ | -------------------------------------------------- | -------- |
| 6:30 PM    | 1      | Instructor Do: Introduction to the Class           | 0:05     |
| 6:35 PM    | 2      | Instructor Do: Dataset Licenses                    | 0:05     |
| 6:40 PM    | 3      | Instructor Do: Dataset Sources                     | 0:05     |
| 6:45 PM    | 4      | Student Do: Explore Dataset Sources                | 0:05     |
| 6:50 PM    | 5      | Instructor Do: Sourcing Data from HTML Tables      | 0:10     |
| 7:00 PM    | 6      | Students Do: Collect HTML Table Data               | 0:15     |
| 7:15 PM    | 7      | Review: Collect HTML Table Data                    | 0:05     |
| 7:20 PM    | 8      | Instructor Do: Intro to APIs and the Client–Server Model | 0:05     |
| 7:25 PM    | 9      | Instructor Do: Requests and JSON                   | 0:15     |
| 7:40 PM    | 10     | Students Do: Introductory API Requests             | 0:20     |
| 8:00 PM    | 11     | Review: Introductory API Requests                  | 0:05     |
| 8:05 PM    | 12     | BREAK                                              | 0:15     |
| 8:20 PM    | 13     | Instructor Do: URL Parameters                      | 0:10     |
| 8:30 PM    | 14     | Student Do: House of Requests                      | 0:20     |
| 8:50 PM    | 15     | Review: House of Requests                          | 0:05     |
| 8:55 PM    | 16     | Instructor Do: JSON to Pandas and Iterative Requests | 0:10     |
| 9:05 PM    | 17     | Student Do: TV Ratings DataFrame                   | 0:15     |
| 9:20 PM    | 18     | Review: TV Ratings DataFrame                       | 0:05     |
| 9:25 PM    | 19     | Instructor Do: End Class and API Key Sign Ups      | 0:05     |
| 9:30 PM    |        | END                                                |          |

---

### 1. Instructor Do: Introduction to the Class (5 min)

Open the slideshow and use the first few slides to facilitate your welcome to the class.

Welcome students and explain that, in today’s lesson, they will focus on how to get started on collecting training data for their AI applications. Remind students that good training data has a direct impact on the performance of AI models, so the groundwork of sourcing the data is an important first step in developing a robust model. Let students know that, in today’s lesson, they will reflect on key data sourcing considerations such as licensing and choosing reliable data sources. They will also begin to collect data from websites using both the Pandas and the HTTP requests libraries.

---

### 2. Instructor Do: Dataset Licenses (5 min)

Continue to use the slideshow to facilitate this conversation.

Use the following points to introduce terms of use:

* Every day, the world we live in becomes more and more data abundant. The volume of data generated on a daily basis is higher than ever before.

* Although data is abundant and easily accessible, there are many important considerations when using data in any project or application. One of these key considerations is the licensing and terms of use for each dataset.

* As is the case with images or information, you need permission from whoever owns the data to use it for your projects. Often, the owners/authors of the data will restrict the permissions for use based on the project you intend to use it for. For example, you may be allowed to use a dataset for a school project or for a personal passion project but may be prohibited from using the same dataset for a for-profit project.

Use the following points to introduce CC licensing:

* The **Creative Commons (CC) licenses** are six different licenses that provide a standardized framework for granting permissions on any body of work. By using one of the CC licenses, creators can make it clear what other people may or may not use their works for.

Use each link to introduce a few of the licenses. Explain to students that we won’t go in depth on every one of the six licenses but will briefly highlight a few here. Be sure to emphasize that these licenses are intricate legal documents and the abbreviated, human-readable versions we will be looking at in the lesson today are no substitute for the full texts:

* **Creative Commons Public Domain Dedication ([CC0](https://creativecommons.org/publicdomain/zero/1.0/))** - Highlight that there are no restrictions on what reusers can do, no attribution required, and no restrictions on how the resulting work should be used or licensed. This license renders a work entirely in the public domain.

* **Creative Commons Attribution ([CC BY](https://creativecommons.org/licenses/by/4.0/))** -  Highlight that there are no restrictions on what the reuser’s work can be used for, as long as the original creator is fairly credited.

* **Creative Commons Attribution-ShareAlike ([CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/))** - Highlight that there are no restrictions on what the reuser’s work can be used for, as long as the original creator is fairly credited and the new work is licensed under the same exact license as the original.

* **Creative Commons Attribution Non-Commercial ([CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/))** - Highlight that this license prohibits commercial use. It also requires that the original creator be fairly credited. Mention that none of the datasets used in this course fall under this license; however, students should keep a sharp eye out for it outside of the course.

Use the following points to introduce OD licensing. Reiterate that these licenses are intricate legal documents and the abbreviated, human-readable versions we will be looking at in the lesson today are no substitute for the full texts:

* **Open Data Commons Open Database License ([ODbL](https://opendatacommons.org/licenses/odbl/summary/))** - Highlight that any works created using the original database must credit the original author, as stipulated in the original database, and that the new works must also be licensed under the ODbL license. Note that any new works must be kept open.

* **Open Data Commons Attribution License ([ODC-By](https://opendatacommons.org/licenses/by/summary/))** - Highlight that any works created using the original database must credit the original author, as stipulated in the original database.

* **Open Data Commons Public Domain Dedication and License ([PDDL](https://opendatacommons.org/licenses/pddl/summary/))** - Highlight that there are no restrictions on new work created using these databases, as they are considered to be in the public domain.

Use the following points to conclude this section:

* Note that you only need to be concerned about licensing and terms of use when working with secondary data&mdash;i.e., data that you didn’t collect and are getting from another party. If you are making data that you collected and own available to others, you should select an appropriate license.  

* Now that we have covered the most pressing regulatory concern when working with datasets, we will move onto another key consideration: sourcing quality data.

---

### 3. Instructor Do: Dataset Sources (5 min)

Continue with the slideshow to facilitate this discussion.

Remind students that datasets are a foundational part of ML applications, so sourcing them is an equally essential skill.

* When engineering AI models, developers don’t just design, build, and run the models. By default, model development is an iterative process, where the model is tested for accuracy and **optimized** accordingly.

Let students know that in modules to come, they will practice optimizing models to improve performance.

Emphasize that unreliable training data could compromise the accuracy of the model’s results

* Since data is such a crucial part of any AI model, it’s important to source the highest possible quality of data that is applicable to your model.

* Anybody who works with data needs to have a healthy amount of skepticism towards any data they receive. In a data-abundant world, it is to be expected that some data is incomplete, invalid, or unreliable.

* It is up to you, as the developer whose work can be impacted by poor data integrity, to scrutinize the integrity of the data you intend to use. This includes being conscious of where you source your data from.

Let students know that there are a few places to source reliable data. The instructor should note that students will have a chance to explore the UCI ML Library and OpenML libraries in the next activity so there’s no need to go into great detail when introducing them.

As you take students through some of the following examples, take a moment to click through to a dataset or locate a page about the terms of use to emphasize the importance of checking this information before downloading, accessing, or using a dataset.

* Some institutions, such as government agencies, publish data that is generally of sound integrity and is freely available. You should still, however, take note of any terms of use before using the datasets.

* [Data.gov](https://data.gov/) is a central resource for government data in the United States, helping you locate federal, state, and local government data. Other governments around the world often also have similar resources, such as [Australia](https://data.gov.au/) and [Singapore](https://data.gov.sg/). Review the [Learn about copyrighted government works](https://www.usa.gov/copyrighted-government-works) page, which may be linked to from datasets available on data.gov.

* There are websites like the [UCI Machine Learning Library](https://archive.ics.uci.edu/)  and [OpenML](https://www.openml.org/) specifically designed for sourcing data for machine learning projects. Both sites provide some important information about how the data was collected, its authors, any academic papers it is linked to, and explicit direction on how to cite the data and which license is applicable.

    * The databases in the UCI Machine Learning Library are listed with information about the attributes of the dataset as well as which subject area it is related to and what associated ML tasks can be performed with it.

    * Similarly, the datasets on the OpenML website contain detailed information about each dataset’s features and characteristics.

* [Kaggle](https://www.kaggle.com/datasets) is a platform where users can upload datasets for other users to retrieve for their own projects. The platform is also commonly used for machine learning competitions. As always, be weary of any restrictions on the use of each dataset. Also point out that sometimes users will upload datasets and list them with an open license when they don't actually have the permission to do so, which is why it should be emphasized that these datasets warrant additional scrutiny to find out more about the original source.

* You will notice that, in each of the above examples, we emphasized the importance of making sure that the usage/licensing restrictions for each dataset allows you to use them for your project. This is a golden rule in data sourcing. If you are in doubt, consult the original source of the dataset. For example, if the data was collected as part of a study, the study may be linked on the Kaggle page and you can follow the link to get more information about the data’s terms of use. If the original source is not directly linked, you can do a Google search to find it instead.

Let students know that, in the activity they’re about to do, they will get an opportunity to locate the terms of use and licensing information of a dataset. Mention that some of the links in the activity are the same as the links in this instructor demo.

Reiterate that, depending on the source of the dataset, its terms of use or licensing permissions may not be immediately obvious. However, the absence of that information does not mean that you are allowed to use it for your project. You can only proceed once you have concrete proof that the project you will use it for complies with the authors’ terms of use.

---

### 4. Student Do: Explore Dataset Sources (5 min)

**Corresponding Activity:** [01-Stu_Dataset_Sources](Activities/01-Stu_Dataset_Sources)

In this activity, students will browse a range of dataset sources to locate datasets, terms and conditions, and licensing information.

---

### 5. Instructor Do: Sourcing Data from HTML Tables (10 min)

**Corresponding Activity:** [02-Ins_Pandas_Read_HTML](Activities/02-Ins_Pandas_Read_HTML)

Use the slides to present the material.

Use the following points to introduce this section:

* In the past two sections, you have developed a checklist for responsibly and reliably sourcing datasets for your projects. First, check that the licensing allows you to use the dataset for your intended purposes. Next, do your best to source quality data from reliable sources.

* These sections have covered databases that already exist in an easily accessible format such as CSV. However, you may want to obtain data from a website such as IMDb or a Wikipedia page. This is known as **web scraping** or simply scraping, which is when you extract data from websites.

* Whilst we won't be covering complex web scraping techniques in this course, there is a Python library students have already been introduced to that provides simplified web scraping from HTML tables. Explain that Pandas has some built-in scraping capabilities.

* Visit the Wikipedia article [List of Australian capital cities](https://en.wikipedia.org/wiki/List_of_Australian_capital_cities) to show students the data table listed in the article. While walking through the activity, remember to scroll to the bottom of the Wikipedia page. Highlight the “Text is available under the Creative Commons Attribution-ShareAlike License 4.0” text. This informs us that we can use the content on the page. Remind students that the golden rule to check for licensing still applies when scraping data. Some websites specifically prohibit web scraping in their terms of use.

Send out and open the solution file for this demonstration to go over the code with students.

* Explain that we can use the `read_html` function in Pandas to parse tabular data from HTML.

* Scroll through the table data that `read_html` collects from the Wikipedia article.  

    ![A screenshot of the table output displayed as a list of tables.](Images/read_html_output_2.png)

* Students may be surprised when multiple sections of data are returned from the list. This may be a good time to ask students for their theories on what datatype is stored in the `tables` variable.

* Print the `type` of the `tables` variable to show that the return value of `read_html` is a list.

    ```python
    # What we get in return is a list of DataFrames for any tabular data that Pandas found.
    type(tables)
    ```

    The output is:

    ```text
    list
    ```

* Explain that although your first instinct may be to expect `tables` to be storing DataFrames, when `read_html` attempts to convert the HTML into DataFrames, it returns **a list of DataFrames**. Hence `type(tables)` returns `list` as an output.

* Tell the students that we can use list indexing to grab a reference to the specific DataFrame that we are interested in.

    ```python
    # We can slice off any of those dataframes that we want using normal indexing.
    df = tables[1]
    df.head()
    ```

    ![A screenshot of the resulting DataFrame.](Images/dataframe_from_list_2.png)

* Explain that we often have to do a lot of data cleaning on these scraped DataFrames. Quickly take the students through examples of setting, dropping, slicing the columns, deleting header rows, and resetting indexes.

* Point out that the city population and state/territory population column headers have square brackets that we’d rather not include. We are able to change the column headings by storing the column headings in a list and using normal list indexing to change the specific headers we want. After this, we use `df.columns` to set the column headers equal to the list we adapted.

    ```python
    cols = list(df.columns)
    cols[2] = "City population"
    cols[3] = "State/territory population"
    df.columns = cols
    df.head()
    ```

    ![A screenshot of the resulting DataFrame.](Images/drop_header_rows_2.png)

* Next, point out that the last column, “Image”, is not useful so we should remove or **drop** it from the DataFrame. Note that the second parameter specified in `data.drop` is `axis`, which specifies whether to drop an index or a column. By default, `data.drop` checks the index axis.

    ```python
    df = df.drop(['Image'], axis=1)
    df.head()
    ```

    ![A screenshot of the resulting DataFrame.](Images/drop_column.png)

* Explain that in our current DataFrame, the index is already set to what we would expect the default to be. However, it is useful to know how to use the `reset_index` method to reset the index to the default numbering.

    ```python
    df = df.reset_index(drop=True)
    df.head()
    ```

    ![A screenshot of the resulting DataFrame.](Images/reset_index.png)

* Finally, show students the DataFrame data for one of the states using `loc`. We specify that we are looking for indices where the “State/territory” column is “New South Wales”. The `loc` method returns the indices that satisfy that condition.

    ```python
    df.loc[df["State/territory"]=="New South Wales"]
    ```

    ![A screenshot of the resulting Pandas DataFrame displaying only the New South Wales row.](Images/newsouthwales.png)

* Finally, explain that we can save the DataFrame as a CSV file. We specify the name of the file we want to save the DataFrame to and we specify `index=False` so that the index numbers are not included in the file. Setting `index=False` is optional. By default, the index numbers are included in the CSV.

    ```python
    df.to_csv("australian_city_data.csv", index=False)
    ```

Explain to students that they will now practice collecting HTML table data in an activity.

---

### 6. Students Do: Collect HTML Table Data (15 min)

**Corresponding Activity:** [03-Stu_Collect_HTML_Table_Data](Activities/03-Stu_Collect_HTML_Table_Data/)

Use the slides to present this activity. Introduce this activity using the following prompts:

* We’ve been discussing how HTML data can be extracted and utilized. In this activity, students will use `read_html` from Pandas to scrape a Wikipedia article. They will then clean the resulting DataFrame and export it to a CSV.

---

### 7. Review: Collect HTML Table Data (5 min)

**Corresponding Activity:** [03-Stu_Collect_HTML_Table_Data](Activities/03-Stu_Collect_HTML_Table_Data/)

* **Note:** Due to the nature of using sources we have no control over, it is possible that the Wikipedia page has been modified since this activity was introduced into the curriculum. Use the following as a guide, depending on the circumstances of the modifications:

    * If the table referenced in the activity has been removed from the page, use this time to discuss issues that can arise when using external sources that we have no control over, such as this example, where a table we wanted to use no longer exists, rendering our code unusable. You may also use a different table found on the page, or a different Wikipedia page, but you will have to update the code for any data cleaning that is needed.

    * If the table still exists but its position on the page has changed, update the table index reference in the solution code. There are two cells where this code will need to be updated:

        ```python
        # Find the correct table
        tables[5]
        ```

        ```python
        # Save the table to a DataFrame
        stats_df = pd.DataFrame(tables[5])
        ```

* Open the Solved notebook, send out the file to students and go through the solution. This activity is very similar to the previous Instructor activity except that the index of the DataFrame of interest is at index `5`.

    ```python
    import pandas as pd

    url = 'https://en.wikipedia.org/wiki/2019_FIFA_Women%27s_World_Cup_statistics'

    # Use Pandas' `read_html` to parse the url
    tables = pd.read_html(url)
    tables

    # Find the correct table
    tables[5]
    ```

    ![A screenshot of the html table depicted as a Pandas DataFrame.](Images/world_cup_df.png)

* Emphasize that retrieving the data is not the end of the activity. Since our intention is to perhaps use this data in an AI application in the future, we must save the data to a DataFrame

    ```python
    # Save the table to a DataFrame
    stats_df = pd.DataFrame(tables[5])
    ```

* Point out that the next step should be to clean up the data so that it contains only the information we want to keep. We're not interested in saving the "Total" row, so let's drop it by referencing its index.

    ```python
    # Drop the "Total" row
    stats_df = stats_df.drop(24)
    stats_df
    ```

* Point out that checking the data types reveals that a few columns are objects. The "Pld" and "D" columns could already be converted to numeric columns since we removed the problematic "Total" row (these cells included text that was used for footnote references in the original source).

* Note that the "GD" and "AGD" columns will need some cleaning in order to be able to convert those columns to numerical values. To clean these columns, we will use `str.replace()` as follows:

    ```python
    # Remove the "+" and replace the "−" with "-" from the "GD" and "AGD" columns
    columns = ["GD", "AGD"]
    for column in columns:
        stats_df[column] = stats_df[column].str.replace("+", "", regex=False)
        stats_df[column] = stats_df[column].str.replace("−", "-", regex=False)
    stats_df
    ```

    ![A screenshot of the Pandas DataFrame with the GD column changes.](Images/world_cup_df_str_replace.png)

* Next, we must convert the columns with numerical values into either integer or float data types.

    ```python
    # Convert the following columns to dtype int
    cols = ["Pld", "D", "GD"]

    for col in cols:
        stats_df[col] = stats_df[col].astype('int')
        
    # Convert the "AGD" column to dtype float
    stats_df["AGD"] = stats_df["AGD"].astype('float')
    stats_df
    ```

* Finally, we export the data to CSV.

    ```python
    # Export as a CSV without the index
    stats_df.to_csv("fifa_stats.csv", index=False)
    ```

Ask the students if they still have any questions.

### 8. Instructor Do: Intro to APIs and the Client–Server Model (5 min)

Now that we have a grasp on HTML data collection, let's dive deeper into another important concept: APIs and the client–server model. Let the students know that, for the rest of this module, we will be focusing on using and interacting with APIs.

Open the lesson slides, move to the "Introduction to APIs section", and highlight the following:

* Application programming interfaces, or APIs, are functions and procedures that enable users to gain access to features and data of an underlying system.

* APIs are used to extract data, play games, connect programs to platforms like AWS, and manage personal finances. We will use them to easily extract data that we would otherwise not be able to access. First, let’s touch briefly on how APIs work.

* APIs act as bridges between different components. You can think of them as old-time telephone operators. A user submits a request (or a **call**) to be connected to another entity. The API interprets the request and transmits the request to the target entity. The user then receives a response. Later in this lesson, you will get to see what these requests and responses look like.

* APIs are made by individual developers like the students, as well as private companies and corporations. Some APIs are free, and others require payment for services. The same golden rule for datasets applies to API requests as well: before you use the data, make sure that your use case complies with the licensing and terms of use of the API.

Briefly introduce students to the World Bank API by reviewing an [API request](https://api.worldbank.org/v2/country/us/indicator/NY.GDP.MKTP.CD?format=json) in a web browser. This API contains data about the GDP of different countries.

Share the [licensing terms](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets) with students. We'll cover the documentation in more detail in the next section.

Now that students know what APIs are and how to execute them, it's time they learn what goes on in the backend when an API request is sent. Students will learn the various components of the client–server model through instructor demonstration.

Navigate to the slides for the client–server model, and highlight the following:

* Explain the client–server model architecture. Indicate that the model encompasses the relationship between clients and servers.

  ![client_server_model.png](Images/client_server_model.png)

* Define the client–server model as a structure that outlines the relationship and flow of communication between two components: a client and a server.

  * A **client** is any tool or application that is used to connect to or communicate with a server. This includes internet browsers, mobile devices, and command-line terminals, to name a few. Clients submit requests to servers, and clients receive responses from servers.

  * A **server** is a computer program, device, or hardware. Servers run some form of application and are tasked with interacting and providing functionality to clients. Servers receive requests from clients, and servers send responses back to clients.

* Inform students that client–server requests are commonly `GET` and `POST` requests. `GET` requests fetch data from servers. `POST` requests transmit data (like user credentials for authorization) to servers.

* Communicate that the client–server model is what handles all traffic and requests sent to a server. This includes websites, APIs, and databases. The client–server model architecture ensures that request calls and tasks made to servers are handled appropriately and effectively. When students perform a search on Google or log in to Facebook.com, they're enacting the client–server model.

---

### 9. Instructor Do: Requests and JSON (15 min)

**Corresponding Activity:** [04-Ins_Requests_JSON](Activities/04-Ins_Requests_JSON)

Open the activity notebook and highlight the following:

* Python offers a `requests` package that can be used to submit API requests through a protocol known as `HTTP`. The `requests` library supports `GET`, `POST`, and `PUT` requests, just to name a few. `GET` requests will be the focus for this class.

* Each type of request serves a different purpose.

  * `GET` requests are used to extract and acquire data from a server.

  * `POST` requests are used to push new or updated data to the server. They can also be used to make requests like `GET`, while including arguments in the header rather than the URL.

  * `PUT` requests are used to overwrite content on the server.

* Remind students that APIs play a key role in analytic data pipelines, often being the source of data or a means to analyze data. By submitting requests in Python, APIs can be used in-line with other processing. For example, if you built an application that makes weather predictions and you wanted to automatically update the weather data as new data becomes available, you could make a daily request to a weather API to fetch the latest data.  

Navigate to the [World Bank API](http://data.worldbank.org/developers) website, and explain that the World Bank API provides data on a variety of topics, including lending types, income levels, and much more.

Open the [Basic Call Structure](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures) link, and explain the documentation's notes on argument-based vs URL-based queries, as captured in the following image:

![Query String vs REST-Style API Calls](Images/WorldBank_Docs.png)

Explain the following:

* Argument-based query strings allow us to include parameters as part of our `GET` request to filter our results, where each parameter is labeled and follows a question mark (`?`) at the end of the base URL.

* REST-based API calls can still use parameters, but they are generally not labeled and must be in the correct position in the URL in order to work.

* Argument-based queries are far more common than URL-based queries.

* Students will have the opportunity to interact with both types of API calls throughout this module.

* It is important to review an APIs documentation in order to correctly format the URL of an API request.

Demonstrate with live code how to use the Python `requests` library, and use the following discussion points:

* The `requests` library has to be imported in order to be used.

    ```python
    import requests
    ```

* The first step to using the requests library after importing it is to declare a variable that will hold the URL.

    ```python
    url = "http://api.worldbank.org/v2/country/us/indicator/NY.GDP.MKTP.CD"
    ```

* Because most APIs support multiple output formats, the next step is to specify the desired output format. This can be added to the URL with a format tag, `?format=`. Common formats used are JSON, CSV, and XML. For this lesson, JSON will be the focus. The format tag will need to be appended to the URL string previously created.

* JSON objects are good to use because their structure is similar to Python dictionaries, which means we can navigate and process them in the same way we would navigate dictionaries. It is common for APIs to output data as JSON objects.

* Ask students if anyone remembers how to append to a string. (Answer: concatenation, which uses `+` to join strings)

    ```python
    url = url + "?format=json"
    ```

* `GET` requests can be sent using the `requests.get` function. The function accepts the request URL as an argument.

    ```python
    # Execute GET request
    requests.get(url)
    ```

    The output is:

    ```text
    <Response [200]>
    ```

* Most APIs incorporate programming that will return code with each server response. These are called **response codes**. A list of common response codes and their meanings can be found below.

    ```text
    Common Response Codes
    100s: Informational
    200s: Success
    300s: Redirection
    400s: Client Error
    500s: Server Error
    ```

* For example, you’ve likely encountered a `404` error when trying to access a webpage. This is a client error indicating that the client couldn’t find the page you requested. 

* When querying APIs, be aware of the `401` response code, returned for unauthorized requests. This means that authentication is required to access the content. In the next lesson, we will learn about how API keys help prevent `401` errors.

* Students may also encounter a `429` response code, which happens when too many requests are being made of a client. Some APIs, like the New York Times APIs, which we'll be covering in the next class, limit the number of requests you're allowed to make in a given timeframe. This will also be covered in more detail in the next class.

* Be aware that a response code in the 200s means the request was successful but not always ideal. For example, if you received a `204` response, it means that the request was successfully fulfilled but that there is no content being returned.

Urge students to google any unfamiliar response codes and then draw their attention back to the demonstration example:

Point out that, in this example, the response was `200`. Ask students to google what the `200` HTTP code means. Answer: “OK”. Explain that this code means we have succeeded in our request.

* Returning to the demo code, demonstrate the following response code results: `404` and `500`. Explain that we can use these results in our code's decision-making. For example, we may want to use an `if-else` statement to check a response code and proceed in our code only if the response is code `200`, or print an error message with the response code if `200` was not returned. If we want different error messages depending on the response code, this could be a good place to use a match statement.

* Point out that the `.text` part of `requests.get(url).text` returns just the text from the response, without any symbols or response code.

* Note that output from a `GET` request is best saved as a variable. This allows for the output to be parsed and manipulated down the road.

    ```python
    # Execute GET request and store response
    response_data = requests.get(url)
    ```

* The actual data returned from the server, called **content**, can be accessed with the `content` attribute.

    ```python
    # Get content
    response_content = response_data.content
    print(response_content)
    ```

    This call returns a length JSON object:

    ![A screenshot depicting the unformatted JSON output.](Images/submit_python_request.png)

* The `json` function from the `json` library can be used to format API output in true JSON format.

    ```python
    import json

    # Format JSON
    data = response_data.json()
    ```

* To improve visual formatting, even more, the `json.dumps` function can be used to add indentations to the JSON data to make the JSON levels and hierarchies more apparent. The `json.dumps` function accepts an argument `indent`, which can be configured to change the indents. `indent=4` is commonly used.

    * Communicate to students that the `json.dumps` function only visually formats the JSON output on the screen; it does not alter the underlying JSON structure.

    * Remind students that coding best practices dictate that code should be kept as readable as possible for the benefit of developers and anybody who may collaborate on the project with them or after them.

        ```python
        # Add indents to JSON and output to screen
        print(json.dumps(data, indent=4))
        ```

        ![A screenshot depicting the JSON output formatted with indentation.](Images/json_with_indent.png)

* JSON data has to be selected based on levels and hierarchies. For example, some JSON objects are organized by JSON object -> JSON array -> attribute. Some have multiple objects, and others have multiple JSON arrays. Either way, accessing JSON data is just like accessing nested data in a dictionary, which you covered in an earlier module. Square brackets are used as **keys** to specify which level and hierarchy should be retrieved.

* In the example below, the first set of square brackets refers to the JSON object. So the `[1]` is referencing the second JSON object. Similarly, the second set of square brackets refers to the second JSON array. Following this, for our `country` variable, we have a third set of square brackets `['country']`, which locates the `country` key nested inside the second JSON array, and the fourth set of square brackets, `['value']`, which specifies which attribute of the `['country']` object should be returned, in this case `value`.

    ```python
    # Select country and GDP value for second row
    country = data[1][1]['country']['value']
    gdp_value = data[1][1]['value']

    print(country)
    print("GDP Value: " + str(gdp_value))
    ```

    ```text
    Country: United States
    GDP Value: 23315080560000
    ```

Ask if there are any remaining questions before moving forward.

---

### 10. Students Do: Introductory API Requests (20 min)

**Corresponding Activity:** [05-Stu_Intro_API_Requests](Activities/05-Stu_Intro_API_Requests/)

In this activity, students will submit `GET` requests using the Python `requests` library for one of the assigned `request urls`. Then they will interpret the JSON output and save a fact or other value from the JSON output as a variable.

---

### 11. Review: Introductory API Requests (5 min)

**Corresponding Activity:** [05-Stu_Intro_API_Requests](Activities/05-Stu_Intro_API_Requests/)

Open the solution file, send it out to students, and go through the code. Be sure to point out the following:

* There were multiple URLs provided. We selected the `star_wars_url` to review, as shown in the following code. If a different URL had been selected, then it would require different references to the JSON nodes in the code.

    ```python
    # Declare "url" variables
    star_wars_url = "https://swapi.dev/api/people/"

    # Execute "GET" request with url
    response_data = requests.get(star_wars_url)

    # Print "response_data" variable
    print(response_data)
    ```

* We have to use the `json()` method on the `response_data` to collect the response in JSON format. Then, we use `json.dumps(data, indent=4)` to print the data in a human readable format. This allows us to easily identify the location of the data we want to collect.

    ```python
    # Format data as JSON
    data = response_data.json()

    # Use json.dumps with argument indent=4 to format data
    print(json.dumps(data, indent=4))
    ```

* Note that this is largely a repeat of the procedure followed in the instructor demo. Focus more time on making sure students understand how to select values from the JSON content.

* In our selected data, the `['results']` node refers to the actual data we want to collect. These results have returned a list of "people," that is, Star Wars characters, and each list item contains a dictionary of information about that character. Item `[0]` refers to the information about Luke Skywalker. From here, we have chosen to save the `name` and `birth_year` fields. If we had selected `[1]`, then we would have saved data about C-3PO.

    ```python
    # Select two values
    selected_value = data['results'][0]['name']
    selected_value_2 = data['results'][0]['birth_year']

    # Print selected values
    print(selected_value)
    print(selected_value_2)
    ```

    The final output is:

    ```text
    Luke Skywalker
    19BBY
    ```

---

### 12. BREAK (15 min)

---

### 13. Instructor Do: URL Parameters (10 min)

**Corresponding Activity:** [06-Ins_URL_Parameters](Activities/06-Ins_URL_Parameters/)

Open the slides to assist with this content, and highlight the following:

Remind students that they’ve now learnt how to query data from an API using a request URL. Introduce the need for having parameters in the request URLs as follows:

* Think about when you search for something on a regular ecommerce site. How often do you find the product listing you’re looking for on the first try? You often have to customize your search by applying filters or sorting the results in a certain way to end up with something that’s useful to you. In the same way, standard URL requests often need to be tweaked to give us the specific information we want.

* To customize our requests, we use **parameters**, which are text-based additions to the basic request URL that hone our request.

* Each API call supports a set of parameters. These parameters can be used to help direct the API toward the data needed or be used to reduce the amount of data being returned by the server.

  * Ask the students: We've already made some API requests using parameters. Can anyone remember any examples?

    **Answer**: When using the `?format=json` tag.

* **Parameters** can be specified in one of two ways. Parameters can follow `/` forward slashes or be specified by parameter name and then by the parameter value.

    ```text
    Parameter provided after /
    http://numbersapi.com/42
    ```

    ```text
    Parameter provided using parameter name and value
    http://numbersapi.com/random?min=10?json
    ```

* When used with parameter names, URL parameters have to be separated from the request URL with the `?` symbol.

    ```text
    http://numbersapi.com/random?min=10
    ```

* Multiple parameters can be passed in with the same URL by separating each parameter with an `&` symbol

    ```text
    http://numbersapi.com/random?min=10&max=20
    ```

Open the solution, and conduct a dry walkthrough of the following solution. Touch on the following discussion points:

* The requests `GET` function can be used to submit a parameterized request to the Numbers API to get trivia facts about the number 42.

    ```python
    import requests
    import json

    # Create parameterized url
    request_url = "http://numbersapi.com/42?json"

    # Submit and format request
    response_data = requests.get(request_url).json()
    print(json.dumps(response_data, indent=4))

    # Select fact
    response_data['text']
    ```

Highlight where the parameters “42” and “?json” appear in the `request_url`. 

Open the starter file, and live code the following:

* The Numbers API URL can be parameterized to execute for the number `8` instead of `42`.

    ```python
    # Create parameterized url
    request_url = "http://numbersapi.com/8?json"

    # Submit and format request
    response_data = requests.get(request_url).json()
    print(json.dumps(response_data, indent=4))

    # Select fact
    response_data['text']
    ```

Point out the parameter “8” in the `request_url` and highlight that this time instead of executing for 42, the API is executed for 8.

Ask the students if they have any remaining questions before moving on.

Now that we’ve covered a few examples of how URL parameters work, let the students know it’s their turn to practice using them in the next activity.

---

### 14. Student Do: House of Requests (20 min)

**Corresponding Activity:** [07-Stu_House_of_Requests](Activities/07-Stu_House_of_Requests/)

This activity is dedicated to giving the students an opportunity to use a fun API. Students play a game of blackjack using the Deck of Cards API. The key skills reinforced in this activity include the execution of GET requests using the Python requests library, extraction of JSON elements, and parameterization of API request URLs.

Students can play the game against a classmate or imaginary dealer. Students are encouraged to work as partners so they can pair-program and play against one another.

---

### 15. Review: House of Requests (5 min)

**Corresponding Activity:** [07-Stu_House_of_Requests](Activities/07-Stu_House_of_Requests/)

Facilitate a dry walkthrough of the solution utilizing the following discussion points:

* Passing parameters to APIs through request URLs gives users the ability to configure and control API actions. By passing parameters to the request URLs for the Deck of Cards API, users can create and shuffle a deck of cards. Parameters also allow users to draw `n` number of cards from the deck.

    ```python
    create_deck_url = f"https://deckofcardsapi.com/api/deck/new/shuffle/?deck_count=6"
    draw_cards_url = f"https://deckofcardsapi.com/api/deck/{deck_id}/draw/?count=2"
    shuffle_deck_url = f"https://deckofcardsapi.com/api/deck/{deck_id}/shuffle/"
    ```

Point out each parameter to the students as demonstrated in the following image. Note: the image is for the instructor’s benefit only and not part of the student-facing material. Do not show it to students. Use the code to point these things out.

  ![An annotated screenshot demonstrating the different parameters in the URLs.](Images/parameters.png)

* The Deck of Cards API has some parameters that need to be specified using a forward slash `/`. There are other parameters that need to be passed using the `?` symbol.

Point out which parameters require a forward slash versus a question mark as demonstrated in this image. Note: the image is for the instructor’s benefit only and not part of the student-facing material. Do not show it to students. Use the code to point these things out.

  ![parameter_formats.png](Images/parameter_formats.png)

* String interpolation is a common way to pass parameters to request URLs. This allows for parameters to be assigned to variables and those variables to be interpolated into the request URLs. This also enables dynamic configuration of parameters and removes instances of hard-coded parameter values.

    ```python
    draw_cards_url = f"https://deckofcardsapi.com/api/deck/{deck_id}/draw/?count=2"
    shuffle_deck_url = f"https://deckofcardsapi.com/api/deck/{deck_id}/shuffle/"
    print(draw_cards_url)
    print(shuffle_deck_url)
    ```

    ```text
    https://deckofcardsapi.com/api/deck/epigy7ynp5yi/draw/?count=2
    https://deckofcardsapi.com/api/deck/epigy7ynp5yi/shuffle/
    ```

* **Parameterized request URLs** are submitted like any other URL: using the `GET` request. Parameters help simplify the amount of data being returned from the output. This makes parsing JSON objects and rows easy, especially since JSON data often has to be iterated. The more parameters used, the less data returned.

  ```python
  # Draw two cards
  drawn_cards = requests.get(draw_cards_url).json()
  ```

* In order to parse JSON data, the structure of the JSON data needs to be understood. JSON data includes parent objects, one or many JSON objects, and elements and attributes for each JSON object. Each of these has to be specified when extracting values from JSON output.

    ```python
    # Select returned card's value and suit (e.g., 3 of clubs)
    player_1_card_1 = drawn_cards['cards'][0]['value'] + " of " + drawn_cards['cards'][0]['suit']
    player_1_card_2 = drawn_cards['cards'][1]['value'] + " of " + drawn_cards['cards'][1]['suit']

    # Print player cards
    print(player_1_card_1)
    print(player_1_card_2)
    ```

    ```text
    3 of HEARTS
    QUEEN of CLUBS
    ```

Explain the parent, child, attribute indexing of the JSON objects as demonstrated in the following image. Note: the image is for the instructor’s benefit only and not part of the student-facing material. Do not show it to students. Use the code to point these things out.

  ![parse_json.png](Images/parse_json.png)

Transition the class into a review session. Ask the following questions:

* In addition to `deck_id`, what other parameter values should be interpolated for the `draw_cards_url`?

  **Answer:** Count.

* If you were to contribute to the Deck of Cards API, what type of features or functionality would you want to enhance or add?

  **Answer:** Game options (e.g., poker, Go Fish, War, etc.).

  **Answer**: Automated dealing based on game type (e.g., poker, Texas Hold'em, etc.)

  **Answer:** Game specific interactions (e.g., playing War compares player cards turn by turn).

  **Answer:** Turn-based gaming.

  **Answer:** Scoring.

* Has URL parameters made APIs more challenging to use or easier to use?

Ask if there are any remaining questions or comments before continuing.

---

### 16. Instructor Do: JSON to Pandas and Iterative Requests (10 min)

**Corresponding Activity:** [08-Ins_JSON_to_Pandas](Activities/08-Ins_JSON_to_Pandas/)

Open the slideshow to accompany the beginning of this demonstration.

Point out that with the APIs used so far, we've been able to retrieve the information we need using single requests.

Explain that sometimes, APIs will only respond to each request with *some* of the information that we need.

* For example, it's common for APIs to send a limited amount of data in response to each call.

* The New York Times API for retrieving articles, for instance, only returns 10 at a time. In this case, if a programmer wanted to retrieve 30 articles, they would have to make 3 API calls.

Explain that API calls can be made *iteratively* by sending `GET` requests out from within a loop.

Point out that an application may want to retrieve a small subset of articles that have non sequential IDs. For example, a user might want to retrieve the posts whose IDs are 3, 89, and 74, respectively.

* It would be wasteful to retrieve all 100 records, take the three that are desired, and throw away the rest. Rather, the application should request *only the articles needed* and nothing more.

* Explain that this can be done by storing the desired IDs in a list and then making an API call inside a loop for each ID in the list.

Open the activity notebook and explain that in this demonstration, we will query the Open Library API for works by author Neil Gaiman.

```python
# Dependencies
import json
import requests
import pandas as pd

# Open library results limit to 50 results per page. 
# Authors documentation: https://openlibrary.org/dev/docs/api/authors
# URL for Neil Gaiman
url = "https://openlibrary.org/authors/OL53305A/works.json"

# Create an empty list to store the responses
response_json = []
```

Point out that since the API limits the number of results to 50 per page, we will have to use a loop to make iterative requests.

```python
# Make a request for 3 pages of results
for x in range(3):
    print(f"Making request number: {x}")

    # Get the results
    post_response = requests.get(url + "?offset=" + str(x * 50)).json()

    # Loop through the "entries" of the results and
    # append them to the response_json list
    for result in post_response["entries"]:
        # Save post's JSON
        response_json.append(result)
```

The output is

```text
Making request number: 0
Making request number: 1
Making request number: 2
```

Ask students why we’ve set the `?offset=` parameter to `str(x * 50). Answer: To make sure that we don’t just get the same 50 works each time we submit the request. Each new request retrieves the next 50 Neil Gaiman works in the database.

```python
# Now we have 150 book objects, 
# which we got by making 3 requests to the API.
print(f"We have {len(response_json)} books!")
```

```text
We have 150 books!
```

Remind students that we can take a peak at what the content looks like:

```python
# preview the JSON
print(json.dumps(response_json, indent=4))
```

This output is lengthy. Refer to the solutions notebook to show students what it looks like.

Draw attention to the first `"key"` in the results:

```text
"key": "/works/OL15168890W"
```

Explain that this `"key"` can be used to make an API request for information about the specific book title. We can use API results to make additional API requests, which students will get to try this themselves in the next class, and will be required to do so in this week's Challenge.

Remind students that retrieving the data is just the beginning. Our actual goal is to put the data to use in some application that we build. The best way to prepare data for this is to convert it into a DataFrame.

```python
# Convert the results to a DataFrame
gaiman_books_df = pd.DataFrame(response_json)
gaiman_books_df
```

The resulting DataFrame is 150 rows x 18 columns. Use the solutions notebook to show students the DataFrame. Scroll across the 18 columns and point out that some of the columns contain nested data.

* We can use the Pandas `json_normalize()` method to automatically create separate columns for each of those nested items.

    ```python
    # Convert the results to a DataFrame, normalizing the JSON
    gaiman_books_normalized_df = pd.json_normalize(response_json)
    gaiman_books_normalized_df
    ```

The resulting DataFrame is 150 rows and 22 columns. In the solutions notebook, point out that the normalization has separated the nested data and introduced more columns. Identify the new columns as: `type.key`, ‘created.type`, `created.value`, `description.type` and `description.value`

---

### 17. Student Do: TV Ratings DataFrame (15 min)

**Corresponding Activity:** [09-Stu_TV_Ratings_DataFrame](Activities/09-Stu_TV_Ratings_DataFrame/)

In this activity, students will create an application that reads in a list of TV shows, makes multiple requests from an API to retrieve information, and creates two Pandas DataFrames.

When introducing the TV ratings application, provide some context around why making requests for this kind of data could be useful in real-world situations that use AI applications. Examples could include in making decisions about recommendations with a machine learning recommendation engine, or studios making decisions around what shows to produce.

---

### 18. Review: TV Ratings DataFrame (5 min)

**Corresponding Activity:** [09-Stu_TV_Ratings_DataFrame](Activities/09-Stu_TV_Ratings_DataFrame/)

Open the solved notebook and go through the code with the class line by line. Make sure to discuss the following points:

* Ask students how data was isolated from each response and loaded into a Pandas DataFrame. **Answer:** In the `for` loop that requests data from the API for each movie, the movie’s title, rating, and content were appended to a list that could then be used to create a DataFrame.

* For students who made their own list of TV shows, ask if any requests failed to return results and consequently caused an error. Let students know that the next lesson will cover exception handling. Exception handling makes it possible for the program to still not run (i.e., not break) even if an error is raised.

* You may wish to review the final column names created by `json_normalize()`. You could do this by opening the exported CSV file to show the fields from the nested objects in the JSON (e.g. the columns that include a period in them like `schedule.time`, `schedule.days`, `rating.average`, `webChannel.id`, `webChannel.name`,and  `webChannel.country`). We can also see these columns using `all_results_df.columns` (which is included in the solution but not required for students).

---

### 19. Instructor Do: End Class and API Key Sign Ups (5 min)

In preparation for the next lesson, let students know that they should sign up for the following APIs:

* [New York Times](https://developer.nytimes.com/accounts/create)
* [TMDB](https://developer.themoviedb.org/reference/intro/getting-started)
* [OpenWeatherAPI](https://openweathermap.org/api)
* [US Census](https://api.census.gov/data/key_signup.html)

Let students know that the instructions for signing up for these accounts are on the Getting Started page on Canvas.

You may choose to make use of office hours to demonstrate how to sign up for these developer accounts. Encourage students to take full advantage of office hours by reminding them that this is their time to ask questions and get assistance from instructional staff as they learn new concepts.

Open the recap slide in the slide deck and let students know that they have reached the end of the lesson. Read through the learning objectives on the recap slide to sum up what students have achieved in this lesson.

Use the following questions to prompt learners to reflect on their learning:

* How has your understanding of data scraping changed or evolved today?
* Do you have any lingering questions about the content?
* What is one thing that you are eager to learn more about or are wondering about?

Use these points as a guide to briefly summarize the next lesson:

* In today’s lesson you began some basic work with APIs. Tomorrow’s lesson will flesh that out further. We will cover some best practices on keeping API keys secure and then practice submitting requests to the New York Times and the OpenWeather APIs.

* By the end of the next lesson students will be able to begin work on the Challenge. The Challenge is all about movies, movies, movies. Students will retrieve data from both the NYT and the TMDB APIs, then combine the data in a DataFrame before cleaning the data and finally exporting to CSV.

---

© 2023 edX Boot Camps LLC. Confidential and Proprietary. All Rights Reserved.
