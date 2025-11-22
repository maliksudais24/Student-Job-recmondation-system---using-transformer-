import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from abc import ABC, abstractmethod
import re
 
class datawragging(ABC):
    def __init__(self,file_path):
        self.file_path = file_path
        self.df = None
        self.load_and_clean_data()

    def load_and_clean_data(self):
        pass
class studentdatavisulaizer(datawragging):
    def load_and_clean_data(self):
        df = pd.read_csv(self.file_path)
        print(df.columns.tolist())

        #cleaning the data and features 
        df.columns= df.columns.str.replace(' ',"",regex=False)
        print(df.columns.tolist())
        df.drop(columns=["Timestamp","Degree","Semester"],inplace=True) 
        # print(df.columns.tolist())

        # applying data wraggling  and eda on each feature 
        #1.selected fileds
        # print(df['SelectedFields'].head(10))
        df['SelectedFields']=df['SelectedFields'].astype(str).str.strip().str.upper()
        df['SelectedFields']=df['SelectedFields'].str.replace(';',',')
        df['SelectedFields']=df['SelectedFields'].str.split(',')
        # print(df['SelectedFields'].isnull().sum())
        # print(df['SelectedFields'].head(10))

     
        #skills
        # print(df['Skills'].head(10)) 
        df['Skills']=df['Skills'].astype(str).str.strip().str.upper()
        df['Skills']=df['Skills'].str.replace(';',',')
        df['Skills']=df['Skills'].str.split(',')
        # print(df['Skills'].isnull().sum())
        # print(df['Skills'].head(10))
        
        # mostlikesubjects
        # print(df['MostLikedSubject'].head(10))
        df['MostLikedSubject']=df['MostLikedSubject'].astype(str).str.strip().str.upper()
        # print(df['MostLikedSubject'].isnull().sum())
        # print(df['MostLikedSubject'].head(10))
        self.subject_map = {
            'DATA SCIENCE': ['DATA SCIENCE', 'DATA SCIENCE ', ' DATA SCIENCE', 'DATA SCIENCE,AI'],
            'AI': ['AI', 'ARTIFICIAL INTELLIGENCE', 'ARTIFICIAL INTELLIGENCE (AI)', 'A.I.'],
            'OOP': ['OOP', 'OBJECT-ORIENTED PROGRAMMING', 'OBJECT ORIENTED PROGRAMMING', 'PF,OOP'],
            'PF': ['PF', 'PROGRAMMING FUNDAMENTAL', 'PROGRAMMING FUNDAMENTALS', 'PROGRAMMING  FUNDAMENTAL'],
            'WEB DEVELOPMENT': ['WEB DEVELOPMENT', 'WEB PROGRAMMING'],
            'DATA STRUCTURE': ['DATA STRUCTURE'],
            'DBMS': ['DATA BASE', 'DATABASE'],
            'UI/UX DESIGN': ['UI/UX DESIGN', 'UI DESIGN'],
            'CYBERSECURITY': ['CYBER SECURITY', 'CYBERSECURITY', 'NETWORK SECURITY', 'CYBER LAWS'],
            'NETWORKING': ['NETWORKING', 'NETWORKONG', 'COMPUTER NETWORKS', 'SWITCH AND ROUTING'],
            'PROGRAMMING': ['PROGRAMMING', 'PROGRAMING', 'PROGRAMMING SUBJECTS'],
            'DLD': ['DLD'],
            'C++': ['C++'],
            'SOFTWARE ENGINEERING': ['SOFTWARE ENGINEERING'],
            'APP DEVELOPMENT': ['APP DEVELOPMENT'],
            'MACHINE LEARNING': ['MACHINE LEARNING'],
            'DATA COMMUNICATION': ['DATA COMMUNICATION'],
            'ENGLISH': ['ENGLISH'],
            'INTRO TO COMPUTER': ['INTRO TO COMPUTER'],
            'GAME PROGRAMMING': ['GAME PROGRAMMING', 'GAME MECHANICS'],
            'DATA VISUALIZATION': ['DATA VISUALIZATION'],
        }
        self.normalized_subjects={}
        for standard,variants in self.subject_map.items():
            for v in variants:
                self.normalized_subjects[v.strip().upper()]=standard
        self.subjectlist =[]
        for row in df['MostLikedSubject'].dropna():
            subjects = re.split(r'\s*AND\s|,|;',row)
            for subject in subjects:
                clean_subjects = subject.strip()
                if clean_subjects:
                   normalized = self.normalized_subjects.get(clean_subjects,clean_subjects)
                   self.subjectlist.append(normalized)
        #  GPA 
        # print(df['GPA'].head(10))
        # print(df['GPA'].isnull().sum())
        df['GPA'] =df['GPA'].str.replace(" ",".")
        df['GPA'] = df['GPA'].str.extract(r"(\d+(\.\d+)?)", expand=True)[0]
        df['GPA']=df['GPA'].astype(float)
        print(df['GPA'].head(10))

        self.df=df
# Save cleaned data
        # self.df.to_csv("clean_data.csv", index=False)
        df['student_text'] = (
            df['SelectedFields'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)) + " " +
            df['Skills'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
        ).str.lower()

        return df

    def visualize_student_field_interset(self):
        field_count = Counter(feild.strip() for feilds in self.df['SelectedFields'].dropna() for feild in feilds if feild.strip()!='')
        plt.figure(figsize=(8, 8))
        plt.pie(field_count.values(), labels=field_count.keys(), autopct='%1.1f%%', startangle=140)
        plt.title('Student Interest in Different Fields')
        plt.tight_layout()
        plt.show()    

    def visualize_most_liked_subjects(self):
        subjects_count =Counter(self.subjectlist)
        subject_count = pd.Series(subjects_count).sort_values(ascending=False)
        subject_count.plot(kind='bar', figsize=(12, 6), color='lightgreen')
        plt.xticks(rotation=45, ha='right')
        plt.title('Most Liked Subjects by Students')
        plt.xlabel('Subjects')
        plt.ylabel('Number of Students')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() 

    def visualize_average_gpa(self):
      avg_gpa = self.df['GPA'].mean()
      plt.figure(figsize=(5, 6))
      plt.bar(['Average GPA'],[avg_gpa],color='blue')
      plt.axhline(avg_gpa, color='red', linestyle='--', label='Avg GPA')
      plt.title('Average GPA by Degree')
      plt.ylabel('GPA')
      plt.xlabel('Degree Program')
      plt.ylim(0, 4.2)
      plt.legend()
      plt.show()
         
    def visualize_skills_distribution(self):
        all_skills = [skill.strip() for skills in self.df['Skills'].dropna() for skill in skills]
        skill_counts = Counter(all_skills)
        plt.figure(figsize=(8, 8))
        plt.pie(skill_counts.values(), labels=skill_counts.keys(), autopct='%1.1f%%', startangle=140)
        plt.title('Skill Distribution Among Students')
        plt.tight_layout()
        plt.show()   

    def summarize_field_and_skill_alignment(self):
       field_counter = Counter([field.strip() for fields in self.df['SelectedFields'] for field in fields])
       skill_counter = Counter([skill.strip() for skills in self.df['Skills'] for skill in skills])
       # Get top 5 fields and skills
       top_fields = field_counter.most_common(5)
       top_skills = skill_counter.most_common(5)
       # Unpack for plotting 
       field_labels, field_values = zip(*top_fields)
       skill_labels, skill_values = zip(*top_skills)
     # Plotting side-by-side bar charts
       fig, axs = plt.subplots(1, 2, figsize=(14, 6))
     # Bar chart for fields
       axs[0].bar(field_labels, field_values, color='lightcoral')
       axs[0].set_title('Top 5 Interested Fields')
       axs[0].set_ylabel('Number of Students')
       axs[0].set_xticklabels(field_labels, rotation=45)
     # Bar chart for skills
       axs[1].bar(skill_labels, skill_values, color='skyblue')
       axs[1].set_title('Top 5 Technical Skills')
       axs[1].set_ylabel('Number of Students')
       axs[1].set_xticklabels(skill_labels, rotation=45)
       plt.tight_layout()
       plt.show()
   
visualizer = studentdatavisulaizer('student_real_data.csv') 
while True:
    print("\nMenu:")
    print("1. Visualize Student Field Interest")
    print("2. Visualize Skills Distribution")
    print("3. Visualize Most Liked Subjects")
    print("4. summarize field and skill of the students")
    print("5.  check for the averge gpa of each department")
    print("6. exist")

    try:
        choice = int(input("Enter your choice (1 to 6): "))
    except ValueError:
        print("Please enter a valid number!")
        continue

    if choice == 1:
        visualizer.visualize_student_field_interset()
    elif choice == 2:
        visualizer.visualize_skills_distribution()
    elif choice == 3:
        visualizer.visualize_most_liked_subjects() 
    elif choice == 4:
        visualizer.summarize_field_and_skill_alignment()   
    elif choice == 5:
        visualizer.visualize_average_gpa()
    elif choice ==6 :    
        break
    else:
        print("Invalid choice. Please select from 1 to 4.")
