from typing import Tuple, Dict


def get_label_map(dataset: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    if dataset == 'ag_news':
        classes = [
            'World',
            'Sports',
            'Business',
            'Sci / Tech'
        ]
    elif dataset == 'cr_sents':
        classes = [
            'benefits',
            'company',
            'responsibility',
            'qualification'
        ]
    elif dataset == 'earning_code':
        classes = ['Stock Units', 'Service Pay', 'Professional Services', 'Referral Bonus', 'Membership', 'Critical Illness', 'Arbitration Award', 'Donations', 'Paid Family Leave', 'Premium Pay', 'Paid Military/Public Safety/Public Service Pay', 'Stock Shares', 'Relocation', 'Standby Pay', 'Paid Annual Leave', 'Paid Maternity Leave', 'Personal Accident', 'Family Care', 'Housing', 'Retirement Savings', 'Disability', 'Termination Payments (excluding Severance)', 'Car Allowance', 'Deferred Compensation - Other', 'Per Diem Pay', 'Group Insurance - Other', 'Paid Parental Leave', 'Loan Recovery', 'Equity - Other', 'Paid Paternity Leave', 'Company Car', 'Transportation', 'Meals', 'Defined Contribution Plan', 'Severance Pay', 'Overtime Pay', 'Electronic Communication', 'Profit Share Pay', 'Fellowship Pay', 'Paid Sabbatical Leave', 'Sign-on Bonus', 'Paid Time Off Pay', 'Perquisites - Other', 'Spot Bonus', 'Life', 'Wellness', 'Pension Payment', 'Paid Jury Leave', 'Paid Bereavement Leave', 'Performance Bonus', 'Regular', 'Lump Sum Payment (including COLA - Cost of Living Adjustment)', 'Uniform', 'Paid Administrative Leave', 'Travel', 'Paid Safe Leave', 'Expense Reimbursement', 'Company Loans', 'Paid Community Service Leave', 'Health', 'Commission Pay', 'Education', 'Paid Sick Leave', 'Stock Purchase Plan', 'Corporate Gift', 'Additional Duty Pay', 'Retention Bonus', 'Stock Options', 'Gratuity Pay']
    elif dataset == 'imdb':
        classes = ['negative', 'positive']
    elif dataset == "medical_specialty":
        classes = ["Allergy / Immunology", "Bariatrics", "Cardiovascular / Pulmonary", "Dentistry", "General Medicine", "Surgery", "Speech - Language", "SOAP / Chart / Progress Notes", "Sleep Medicine", "Rheumatology", "Radiology", "Psychiatry / Psychology", "Podiatry", "Physical Medicine - Rehab", "Pain Management", "Orthopedic", "Ophthalmology", "Office Notes", "Neurosurgery", "Neurology", "Nephrology", "Letters", "Lab Medicine - Pathology", "IME-QME-Work Comp etc.", "Hematology - Oncology", "Gastroenterology", "ENT - Otolaryngology", "Endocrinology", "Discharge Summary", "Dermatology", "Cosmetic / Plastic Surgery", "Consult - History and Phy.", "Chiropractic", "Autopsy"]
    else:
        raise Exception("Dataset not supported: ", dataset)

    label_map = {k: v for v, k in enumerate(classes)}
    rev_label_map = {v: k for k, v in label_map.items()}

    return label_map, rev_label_map