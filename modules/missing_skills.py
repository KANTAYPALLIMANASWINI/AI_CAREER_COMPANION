# ✅ missing_skills.py

# Predefined job role skill requirements
job_skills = {
    'Frontend Developer': ['html', 'css', 'javascript', 'react'],
    'Backend Developer': ['python', 'django', 'sql'],
    'Data Analyst': ['python', 'excel', 'sql', 'data analysis'],
    'Machine Learning Engineer': ['python', 'ml', 'numpy', 'pandas'],
    'Full Stack Developer': ['html', 'css', 'javascript', 'python', 'django'],
    'Software Engineer': ['java', 'oops', 'dsa'],
    'UI/UX Designer': ['figma', 'adobe', 'creativity'],
    'Communication Executive': ['communication', 'english', 'presentation'],
    'DevOps Engineer': ['docker', 'kubernetes', 'aws', 'ci/cd'],
    'Cloud Engineer': ['aws', 'azure', 'gcp', 'terraform'],
    'Cybersecurity Analyst': ['network security', 'threat detection', 'firewall'],
    'Business Analyst': ['excel', 'power bi', 'sql', 'market analysis'],
    'Data Scientist': ['python', 'machine learning', 'data visualization', 'statistics'],
    'AI Research Intern': ['deep learning', 'nlp', 'python', 'transformers'],
    'Game Developer': ['unity', 'c#', '3d modeling', 'game physics'],
    'Mobile App Developer': ['flutter', 'dart', 'android', 'ios'],
    'Technical Writer': ['markdown', 'documentation', 'api writing'],
    'QA Tester': ['manual testing', 'selenium', 'jira', 'bug tracking'],
    'Digital Marketing Executive': ['seo', 'sem', 'google analytics', 'content marketing'],
    'System Administrator': ['linux', 'shell scripting', 'networking'],
    'Blockchain Developer': ['solidity', 'ethereum', 'web3.js'],
    'Embedded Systems Engineer': ['c', 'embedded c', 'microcontrollers'],
    'Robotics Engineer': ['ros', 'python', 'robot kinematics'],
}

# Learning links
learning_links = {
    'python': 'https://www.learnpython.org/',
    'html': 'https://developer.mozilla.org/en-US/docs/Web/HTML',
    'css': 'https://developer.mozilla.org/en-US/docs/Web/CSS',
    'javascript': 'https://www.javascript.info/',
    'react': 'https://react.dev/learn',
    'django': 'https://developer.mozilla.org/en-US/docs/Learn/Server-side/Django',
    'sql': 'https://www.codecademy.com/learn/learn-sql',
    'excel': 'https://exceljet.net/',
    'java': 'https://www.w3schools.com/java/',
    'oops': 'https://www.geeksforgeeks.org/object-oriented-programming-oops-concept-in-java/',
    'dsa': 'https://www.geeksforgeeks.org/data-structures/',
    'communication': 'https://www.coursera.org/learn/wharton-communication',
    'teamwork': 'https://www.edx.org/course/teamwork-skills-communicating-effectively-in-groups',
    'data analysis': 'https://www.coursera.org/specializations/data-analysis',
    'ml': 'https://developers.google.com/machine-learning/crash-course',
    'numpy': 'https://numpy.org/learn/',
    'pandas': 'https://pandas.pydata.org/docs/getting_started/index.html',
    'docker': 'https://docker-curriculum.com/',
    'kubernetes': 'https://kubernetes.io/docs/home/',
    'aws': 'https://aws.amazon.com/training/',
    'ci/cd': 'https://www.redhat.com/en/topics/devops/what-is-ci-cd',
    'azure': 'https://learn.microsoft.com/en-us/training/azure/',
    'gcp': 'https://cloud.google.com/training',
    'terraform': 'https://developer.hashicorp.com/terraform/tutorials',
    'network security': 'https://www.coursera.org/specializations/network-security',
    'threat detection': 'https://www.coursera.org/learn/threat-detection',
    'firewall': 'https://www.geeksforgeeks.org/introduction-to-firewall/',
    'power bi': 'https://learn.microsoft.com/en-us/power-bi/',
    'market analysis': 'https://www.investopedia.com/terms/m/market-analysis.asp',
    'data visualization': 'https://www.tableau.com/learn/training',
    'statistics': 'https://www.khanacademy.org/math/statistics-probability',
    'deep learning': 'https://www.deeplearning.ai/',
    'nlp': 'https://www.analyticsvidhya.com/blog/2021/06/nlp-for-beginners/',
    'transformers': 'https://huggingface.co/learn/nlp-course/chapter1',
    'unity': 'https://learn.unity.com/',
    'c#': 'https://learn.microsoft.com/en-us/dotnet/csharp/',
    '3d modeling': 'https://www.blender.org/learn/',
    'game physics': 'https://www.gamedeveloper.com/programming/game-physics-resources',
    'flutter': 'https://flutter.dev/docs',
    'dart': 'https://dart.dev/guides',
    'android': 'https://developer.android.com/courses',
    'ios': 'https://developer.apple.com/learn/curriculum/',
    'markdown': 'https://www.markdownguide.org/',
    'documentation': 'https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/',
    'api writing': 'https://idratherbewriting.com/learnapidoc/',
    'manual testing': 'https://www.geeksforgeeks.org/manual-testing/',
    'selenium': 'https://www.selenium.dev/documentation/',
    'jira': 'https://www.atlassian.com/software/jira/guides',
    'bug tracking': 'https://www.guru99.com/bug-life-cycle.html',
    'seo': 'https://moz.com/beginners-guide-to-seo',
    'sem': 'https://www.wordstream.com/search-engine-marketing',
    'google analytics': 'https://analytics.google.com/analytics/academy/',
    'content marketing': 'https://www.hubspot.com/resources/ebooks/content-marketing',
    'linux': 'https://linuxjourney.com/',
    'shell scripting': 'https://linuxconfig.org/bash-scripting-tutorial-for-beginners',
    'networking': 'https://www.geeksforgeeks.org/computer-network-tutorials/',
    'solidity': 'https://soliditylang.org/',
    'ethereum': 'https://ethereum.org/en/developers/',
    'web3.js': 'https://web3js.readthedocs.io/en/v1.7.4/',
    'embedded c': 'https://www.geeksforgeeks.org/introduction-to-embedded-c/',
    'microcontrollers': 'https://www.ni.com/en-us/innovations/microcontrollers.html',
    'ros': 'https://emanual.robotis.com/docs/en/platform/turtlebot3/ros_setup/',
    'robot kinematics': 'https://www.udacity.com/course/robot-kinematics--ud1201'
}

# 🔍 Step 1: Find missing skills based on role
def find_missing_skills(user_skills, predicted_role):
    required = job_skills.get(predicted_role, [])
    missing = list(set(required) - set(user_skills))
    return missing

# 📚 Step 2: Get learning links for the missing ones
def get_learning_links(skills):
    links = {}
    for skill in skills:
        links[skill] = learning_links.get(skill.lower(), '🔍 Link not found')
    return links
